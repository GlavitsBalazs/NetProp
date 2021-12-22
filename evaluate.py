# Copyright (C) 2021 Balázs Róbert Glávits

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# long with this program.  If not, see <https://www.gnu.org/licenses/>.

import csv
import functools
import glob
import io
import os
import random
import sqlite3
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence, Optional, Tuple, Iterable

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import tqdm

import netprop


def create_networks():
    if not os.path.exists("gene_network.npz"):
        netprop.create_gene_graph("gene_network.npz")
    if not os.path.exists("shuffled_gene_network.npz"):
        adjacency, node2label, label2node = netprop.load_graph("gene_network.npz", labels=True)
        netprop.shuffle_graph(adjacency, iterations=(adjacency.nnz - 1) * 4, disable_tqdm=False)
        while True:
            comp, _ = sp.csgraph.connected_components(adjacency)
            if comp == 1:
                break
            netprop.shuffle_graph(adjacency, iterations=int(adjacency.nnz * 0.1), disable_tqdm=False)
        netprop.save_graph("shuffled_gene_network.npz", adjacency, node2label)
    if not os.path.exists('permutations'):
        adjacency, node2label, label2node = netprop.load_graph("gene_network.npz", labels=True)
        os.mkdir('permutations')
        count = 100
        files = [os.path.join('permutations', f"permutation_{n:0{int(np.ceil(np.log10(count)))}}.npz")
                 for n in range(count)]
        netprop.create_permutations(adjacency, files, worker_processes=8, iterations=4 * adjacency.shape[0])


def plot_networks():
    def matrix_permute(mat: sp.spmatrix, perm_row: np.ndarray, perm_col: np.ndarray = None):
        if perm_col is None:
            perm_col = perm_row
        m, n = mat.shape
        perm_row_mat = sp.coo_matrix((np.ones(n, dtype=mat.dtype), (np.arange(n, dtype=perm_row.dtype), perm_row)))
        perm_col_mat = sp.coo_matrix((np.ones(m, dtype=mat.dtype), (perm_col, np.arange(m, dtype=perm_col.dtype))))
        return perm_row_mat.tocsr() @ mat @ perm_col_mat.tocsr()

    def reverse_cuthill_mckee(adj: sp.csr_matrix, node_labels: Sequence, symmetric_mode=True):
        permutation = sp.csgraph.reverse_cuthill_mckee(adj, symmetric_mode)
        new_adj = matrix_permute(adj, permutation, permutation)
        new_labels = np.array([node_labels[p] for p in permutation])
        small_n = np.max(new_adj.indices) + 1
        new_adj._shape = (small_n, small_n)
        new_adj.indptr = new_adj.indptr[:small_n + 1]
        new_adj.sort_indices()
        new_labels = new_labels[:small_n + 1]
        return new_adj, new_labels

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6 * 2, 4 * 2), dpi=141.21 / 1.5)
    adjacency = netprop.load_graph("gene_network.npz")
    shuf_adjacency = netprop.load_graph("shuffled_gene_network.npz")
    shuf_adjacency, _ = reverse_cuthill_mckee(shuf_adjacency, range(shuf_adjacency.shape[0]))
    ax1.spy(adjacency, markersize=0.01, rasterized=True)
    ax2.spy(shuf_adjacency, markersize=0.01, rasterized=True)
    plt.savefig('adjacency.pdf', bbox_inches='tight')
    plt.show()


def download_gene_names(output_path):
    """Download the table relating NHCI Entrez gene names to their Ensembl IDs from ensembl.org. Store it in SQLite."""
    query_xml = \
        '<?xml version="1.0" encoding="UTF-8"?>' \
        '<!DOCTYPE Query>' \
        '<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="0" datasetConfigVersion="0.6">' \
        '<Dataset name="hsapiens_gene_ensembl" interface="default">' \
        '<Attribute name="ensembl_gene_id"/>' \
        '<Attribute name="external_gene_name"/>' \
        '<Attribute name="external_synonym"/>' \
        '</Dataset>' \
        '</Query>'
    url = 'https://www.ensembl.org/biomart/martservice?query=' + urllib.parse.quote(query_xml)
    entries: set[Tuple[int, str]] = set()
    with urllib.request.urlopen(url) as tsv:
        for row in csv.DictReader(io.TextIOWrapper(tsv), delimiter='\t'):
            gene_ensembl_id = int(row['Gene stable ID'][4:])
            gene_name = row['Gene name']
            gene_synonym = row['Gene Synonym']
            if gene_name:
                entries.add((gene_ensembl_id, gene_name))
            if gene_synonym:
                entries.add((gene_ensembl_id, gene_synonym))

    if os.path.exists(output_path):
        os.remove(output_path)
    with sqlite3.connect(output_path) as sqlcon:
        sqlcur = sqlcon.cursor()
        sqlcur.executescript('''
        CREATE TABLE "geneNames" (
            "geneEnsemblID"	INTEGER NOT NULL,
            "geneName" TEXT NOT NULL
        );
        CREATE INDEX gene_Name2EnsemblID ON "geneNames"("geneName");
        ''')
        sqlcon.commit()
        sqlcur.executemany('INSERT INTO geneNames VALUES (?, ?)', entries)
        sqlcon.commit()


def find_gene_disease_associations(score_threshold, disgenet_db_path,
                                   gene_names_db_path) -> Iterable[Tuple[str, set[str]]]:
    """
    :return: Iterable of disease UMLS CUI (Unified Medical Language System Concept Unique Identifier)
    and set of associated genes by Ensembl ID with at least `score_threshold` DisGeNet score.
    """
    if not os.path.exists(disgenet_db_path):
        raise FileNotFoundError(f"No such file or directory: '{disgenet_db_path}'. "
                                f"Please download the DisGeNET SQLite database (v7.0) from "
                                f"https://www.disgenet.org/downloads")
    if not os.path.exists(gene_names_db_path):
        download_gene_names(gene_names_db_path)
    with sqlite3.connect(':memory:') as conn:
        conn.execute(f'ATTACH DATABASE "{disgenet_db_path}" as disgenet;')
        conn.execute(f'ATTACH DATABASE "{gene_names_db_path}" as gene_names;')
        sqlcur = conn.cursor()

        diseases = list(sqlcur.execute('''
        SELECT diseaseNID, diseaseId
        FROM (
            SELECT diseaseNID, diseaseId, COUNT(DISTINCT geneNID) AS geneCount
            FROM disgenet.geneDiseaseNetwork JOIN disgenet.diseaseAttributes USING (diseaseNID)
            WHERE score > ?
            GROUP BY diseaseNID
            ORDER BY geneCount DESC
        ) WHERE geneCount > 1;
        ''', (score_threshold,)))
        for disease, diseaseId in diseases:
            res = sqlcur.execute('''
            SELECT geneEnsemblID
            FROM (
               SELECT geneName, MAX(score) as geneScore
               FROM disgenet.geneDiseaseNetwork JOIN disgenet.geneAttributes USING (geneNID)
               WHERE diseaseNID = ?
               GROUP BY geneName
            ) JOIN gene_names.geneNames USING (geneName)
            WHERE geneScore > ?
            GROUP BY geneEnsemblID;
            ''', (disease, score_threshold))
            associated = {f"ENSG{gene:011}" for gene, in res}
            if len(associated) > 0:
                yield diseaseId, associated


def run_loocv(raw_result_file, network_file, prop: netprop.NetworkPropagation,
              permutation_paths: Optional[Sequence[str]], test_case_density=0.1):
    """
    Leave one out cross validation.

    Load DisGeNet and for all diseases find a set of associated genes. For each gene set and each gene, a test case is
    defined with the goal of finding that particular gene given the other genes from the set as input. Only a random
    selection of test cases will be executed, the number of which is approximately the `test_case_density` fraction
    of the number of all available test cases from DisGeNet. The results are stored in a "raw results" numpy file for
    later analysis.
    """
    adjacency, node2label, label2node = netprop.load_graph(network_file, labels=True)

    genes_in_network = set(node2label)
    node_sets = dict()
    for disease, genes in find_gene_disease_associations(score_threshold=0.3, disgenet_db_path='disgenet_2020.db',
                                                         gene_names_db_path='gene_names.db'):
        nodes = {label2node[g] for g in genes if g in genes_in_network}
        if len(nodes) < 2:
            continue
        node_sets[disease] = nodes

    weight_vectors = []
    test_diseases = []
    test_removed_nodes = []
    for disease, nodes in node_sets.items():
        for removed_node in nodes:
            if random.random() > test_case_density:
                continue
            weights = np.zeros(adjacency.shape[0])
            for remaining in nodes:
                if remaining != removed_node:
                    weights[remaining] = 1
            weight_vectors.append(weights)
            test_diseases.append(disease)
            test_removed_nodes.append(removed_node)

    score_vectors, significance_vectors = netprop.evaluate(weight_vectors, prop, adjacency, permutation_paths,
                                                           pool_processes=8, memory_limit=2 * 10 ** 9)
    fields = dict()
    fields['node2label'] = node2label
    fields['weight_vectors'] = weight_vectors
    fields['score_vectors'] = score_vectors
    fields['significance_vectors'] = significance_vectors
    fields['test_diseases'] = test_diseases
    fields['test_removed_nodes'] = test_removed_nodes
    np.savez_compressed(raw_result_file, **fields)


@dataclass
class LoocvResult:
    disease_cui: str
    removed_gene: str
    starting_genes: int
    rank: Optional[int]
    found: int


def read_loocv_results(raw_result_file, significance_level=1.0) -> list[LoocvResult]:
    npz = np.load(raw_result_file, allow_pickle=False)
    node2label = npz['node2label']
    weight_vectors = npz['weight_vectors']
    score_vectors = npz['score_vectors']
    significance_vectors = npz['significance_vectors']
    test_diseases = npz['test_diseases']
    test_removed_nodes = npz['test_removed_nodes']

    results = []
    for i in tqdm.trange(len(weight_vectors)):
        ranking = netprop.node_ranking(weight_vectors[i], score_vectors[i], significance_vectors[i], significance_level)
        result = LoocvResult(disease_cui=test_diseases[i], removed_gene=node2label[test_removed_nodes[i]],
                             starting_genes=np.count_nonzero(weight_vectors[i]) + 1, rank=None, found=len(ranking))
        if test_removed_nodes[i] in ranking:
            result.rank = ranking.index(test_removed_nodes[i]) + 1
        results.append(result)
    return results


def precision_recall(evaluation: Sequence[LoocvResult]):
    max_cutoff = max(r.found for r in evaluation)
    actual_positives = len(evaluation)
    curve = []
    for cutoff in range(1, max_cutoff + 1):
        true_positives = sum(1 for r in evaluation if r.rank is not None and r.rank <= cutoff)
        predicted_positives = sum(min(cutoff, r.found) for r in evaluation)
        curve.append((true_positives / actual_positives, true_positives / predicted_positives))
    return curve


def mean_average_precision(evaluation: Sequence[LoocvResult]):
    """https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision"""
    return np.mean([1 / r.rank if r.rank is not None else 0 for r in evaluation])


def average_precision_score(prc: Sequence[Tuple[float, float]]):
    result = 0
    for i in range(len(prc) - 1):
        a, b = prc[i], prc[i + 1]
        result += (b[0] - a[0]) * b[1]
    return result


def auc(points: Sequence[Tuple[float, float]]):
    """Area under curve by trapezoid method."""
    result = 0
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        result += (b[0] - a[0]) * (min(b[1], a[1]) + abs(b[1] - a[1]) / 2)
    return result


def per_disease_scores(evaluation: Sequence[LoocvResult]):
    group_by_disease = defaultdict(list)
    for res in evaluation:
        group_by_disease[res.disease_cui].append(res)
    with open("evaluation.csv", "wt") as f:
        writer = csv.writer(f)
        for d, res in tqdm.tqdm(group_by_disease.items()):
            prc = precision_recall(res)
            writer.writerow((d, len(res), group_by_disease[d][0].starting_genes, mean_average_precision(res),
                             average_precision_score(prc), auc(prc)))


def parameter_test(alpha, test_case_density=0.05, slow_ones=True):
    random.seed(123)
    run_loocv(f"loocv_rwr@{alpha}.npz", "gene_network.npz",
              permutation_paths=None, test_case_density=test_case_density,
              prop=functools.partial(netprop.rwr_left_similarity, alpha=alpha, solver=sp.linalg.lgmres))

    random.seed(123)
    run_loocv(f"loocv_rwr_right@{alpha}.npz", "gene_network.npz",
              permutation_paths=None, test_case_density=test_case_density,
              prop=functools.partial(netprop.rwr_right_similarity, alpha=alpha, solver=sp.linalg.lgmres))

    random.seed(123)
    run_loocv(f"loocv_rwr_kernel@{alpha}.npz", "gene_network.npz",
              permutation_paths=None, test_case_density=test_case_density,
              prop=functools.partial(netprop.rwr_kernel, alpha=alpha, solver=sp.linalg.lgmres))

    if slow_ones:
        random.seed(123)
        run_loocv(f"loocv_rct_kernel@{alpha}.npz", "gene_network.npz",
                  permutation_paths=None, test_case_density=test_case_density,
                  prop=functools.partial(netprop.rct_kernel, alpha=alpha, solver=sp.linalg.lgmres))

        random.seed(123)
        run_loocv(f"loocv_lap_diff@{alpha}.npz", "gene_network.npz",
                  permutation_paths=None, test_case_density=test_case_density,
                  prop=functools.partial(netprop.laplacian_diffusion_kernel, alpha=alpha))

        random.seed(123)
        run_loocv(f"loocv_diff@{alpha}.npz", "gene_network.npz",
                  permutation_paths=None, test_case_density=test_case_density,
                  prop=functools.partial(netprop.diffusion_kernel, alpha=alpha))


def plot_parameter_test(run=False):
    if run:
        parameter_test(0.1)
        parameter_test(0.6)
        parameter_test(0.9)

        parameter_test(0.05, slow_ones=False)
        parameter_test(0.3, slow_ones=False)
        parameter_test(0.5, slow_ones=False)
        parameter_test(0.8, slow_ones=False)
        parameter_test(0.95, slow_ones=False)

    def evaluate_parameter_test(paths_glob, fmt=''):
        points = []
        for f, alpha in [(f, float(f.split('@')[1][:-4])) for f in glob.glob(paths_glob)]:
            res = read_loocv_results(f)
            aps = average_precision_score(precision_recall(res))
            points.append((alpha, aps))
        points.sort()
        alphas, scores = np.array(points).transpose()
        plt.plot(alphas, scores, fmt)

    plt.figure(figsize=(6, 4), dpi=141.21)
    evaluate_parameter_test("loocv_rwr@*.npz", '+-')
    evaluate_parameter_test("loocv_rwr_right@*.npz", 'x-')
    evaluate_parameter_test("loocv_rwr_kernel@*.npz", '*-')
    evaluate_parameter_test("loocv_rct_kernel@*.npz", 'o-')
    evaluate_parameter_test("loocv_lap_diff@*.npz", 'v-')
    evaluate_parameter_test("loocv_diff@*.npz", '^-')
    plt.xlabel('alfa')
    plt.ylabel('átlagos precizitás')
    plt.legend(['RWR', 'jobb RWR', 'szimmetrikus RWR', 'RCT kernel', 'Laplace diffúziós kernel',
                'exponenciális diffúziós kernel'],
               bbox_to_anchor=(1, 1), loc='upper left')
    plt.xlim(0, 1)
    plt.grid(linestyle='--')
    plt.savefig('parameters.pdf', bbox_inches='tight')
    plt.show()


def plot_size_precision_correlation(run=False):
    if run:
        prop = functools.partial(netprop.rwr_left_similarity, alpha=0.6, solver=sp.linalg.lgmres)
        run_loocv("loocv_full.npz", "gene_network.npz", prop, permutation_paths=None, test_case_density=1)
    evaluation = read_loocv_results("loocv_full.npz")
    sizes, scores = [], []
    group_by_disease = defaultdict(list)
    for res in evaluation:
        group_by_disease[res.disease_cui].append(res)
    for d, res in group_by_disease.items():
        sizes.append(group_by_disease[d][0].starting_genes)
        scores.append(average_precision_score(precision_recall(res)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), dpi=141.21)
    ax1.loglog()
    ax1.scatter(sizes, scores, s=0.5)
    ax1.axhline(0.015, linestyle='--', color='grey')
    ax1.axvline(30, linestyle='--', color='grey')
    # ax1.set_xlabel('gének száma')
    ax1.set_ylabel('átlagos precizitás')
    ax2.scatter(sizes, scores, s=0.5)
    ax2.axhline(0.015, linestyle='--', color='grey')
    ax2.axvline(30, linestyle='--', color='grey')
    ax2.set_xlabel('gének száma')
    ax2.set_ylabel('átlagos precizitás')
    plt.savefig('gene_sets2.pdf', bbox_inches='tight')
    plt.show()


def plot_rwr(run=False, small=False):
    if run:
        prop = functools.partial(netprop.rwr_left_similarity, alpha=0.6, solver=sp.linalg.lgmres)
        permutation_paths = list(glob.glob(os.path.join('permutations', 'permutation_*.npz')))
        permutation_paths.sort()
        permutation_paths = permutation_paths[:100]
        random.seed(123)
        run_loocv("loocv_shuffled.npz", "shuffled_gene_network.npz", prop, None, test_case_density=0.2)
        random.seed(123)
        run_loocv("loocv_rwr06_perm.npz", "gene_network.npz", prop, permutation_paths, test_case_density=0.2)
    res_shuffled = read_loocv_results("loocv_shuffled.npz", significance_level=1.00)
    res = read_loocv_results("loocv_rwr06_perm.npz", significance_level=1.00)
    res_05 = read_loocv_results("loocv_rwr06_perm.npz", significance_level=0.05)

    if small:
        res_shuffled = [r for r in res_shuffled if r.starting_genes <= 30]
        res = [r for r in res if r.starting_genes <= 30]
        res_05 = [r for r in res_05 if r.starting_genes <= 30]
        print(len(res))

    prc = precision_recall(res)
    prc_05 = precision_recall(res_05)
    prc_shuf = precision_recall(res_shuffled)
    print(average_precision_score(prc))
    print(average_precision_score(prc_05))
    print(average_precision_score(prc_shuf))

    fig = plt.figure(1, figsize=(6, 4), dpi=141.21)
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.plot(*np.array(prc).transpose())
    ax1.plot(*np.array(prc_05).transpose())
    ax1.plot(*np.array(prc_shuf).transpose())

    ax1.legend(['RWR', 'szűrt RWR', 'véletlenszerű osztályozó'])
    ax1.set_xlabel('szenzitivitás')
    ax1.set_ylabel('precizitás')
    ax1.grid(linestyle='--')
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.savefig('rwr_small.pdf' if small else 'rwr.pdf', bbox_inches='tight')
    plt.show()
    plt.show()

    fig = plt.figure(1, figsize=(6, 4), dpi=141.21)
    ax2 = fig.add_subplot(1, 1, 1)
    counts = np.bincount([r.rank for r in res])
    dist = np.cumsum(counts)
    dist = dist.astype(float) / dist.max()
    ax2.plot(dist)

    counts = np.bincount([r.rank if r.rank is not None else len(counts) + 1 for r in res_05])
    dist = np.cumsum(counts)
    dist = dist.astype(float) / dist.max()
    ax2.plot(dist)

    counts = np.bincount([r.rank for r in res_shuffled])
    dist = np.cumsum(counts)
    dist = dist.astype(float) / dist.max()
    ax2.plot(dist)

    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, 6000)
    ax2.set_xlabel('sorszám')
    ax2.set_ylabel('kumulatív gyakoriság')
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.legend(['RWR', 'szűrt RWR', 'véletlenszerű osztályozó'])
    ax2.grid(linestyle='--')

    plt.savefig('rwr_small_cum.pdf' if small else 'rwr_cum.pdf', bbox_inches='tight')
    plt.show()
    ax2.set_ylim(auto=True)
    ax2.set_xlim(0, 250)
    plt.savefig('rwr_small_cum2.pdf' if small else 'rwr_cum2.pdf', bbox_inches='tight')
    plt.show()


def cumulative_ranks_plot():
    results = read_loocv_results("loocv_rwr06_perm.npz", significance_level=1.00)
    counts = np.bincount([r.rank for r in results])
    dist = np.cumsum(counts)
    dist = dist.astype(float) / dist.max()
    plt.figure(figsize=(6, 4), dpi=141.21)
    plt.plot(dist)
    plt.ylim(0, 1)
    plt.xlim(0, 100)
    plt.xlabel('sorszám')
    plt.ylabel('kumulatív gyakoriság')
    plt.savefig('rankfreq_sq.pdf', bbox_inches='tight')
    plt.show()
