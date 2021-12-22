#!/usr/bin/env python3

# Copyright (C) 2021  Balázs Róbert Glávits

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

import argparse
import csv
import functools
import glob
import gzip
import io
import itertools
import os
import pathlib
import random
import sys
import urllib.parse
import urllib.request
from multiprocessing import Pool
from typing import Callable, Tuple, Optional, Sequence, Iterable, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import tqdm
from scipy.sparse.linalg import LinearOperator


def save_graph(output_file, adjacency: sp.spmatrix, node_labels: Optional[Sequence] = None):
    """
    Save the simple graph adjacency matrix.
    The format is similar as what `scipy.sparse.save_npz` does with `scipy.sparse.csr_matrix` input
    but add the node labels too.
    """
    upper: sp.csr_matrix = sp.triu(adjacency, format='csr')  # upper compresses better than lower
    upper.indices = upper.indices.astype(np.min_scalar_type(upper.shape[0]))
    if node_labels is not None:

        fields = dict()
        fields['labels'] = node_labels
        fields['data'] = upper.data
        fields['indices'] = upper.indices
        fields['indptr'] = upper.indptr
        fields['format'] = upper.format.encode('ascii')
        fields['shape'] = upper.shape
        np.savez_compressed(output_file, **fields)
    else:
        sp.save_npz(output_file, upper)


def load_graph(input_file, labels=False) -> Union[sp.csr_matrix, Tuple[sp.csr_matrix, Sequence, dict]]:
    if labels:
        npz = np.load(input_file, allow_pickle=False)
        upper = sp.csr_matrix((npz['data'], npz['indices'], npz['indptr']), shape=npz['shape'])
        node2label = npz['labels']
        label2node = dict()
        for node, label in enumerate(node2label):
            label2node[label] = node
        adjacency = upper + upper.T
        return adjacency, node2label, label2node
    else:
        upper = sp.load_npz(input_file)
        adjacency = upper + upper.T
        return adjacency


def create_gene_graph(output_file, quiet=False, protein_graph_path='9606.protein.links.v11.5.txt.gz', cutoff_score=700):
    def matrix_permute(mat: sp.spmatrix, perm_row: np.ndarray, perm_col: np.ndarray = None):
        if perm_col is None:
            perm_col = perm_row
        m, n = mat.shape
        perm_row_mat = sp.coo_matrix((np.ones(n, dtype=mat.dtype), (np.arange(n, dtype=perm_row.dtype), perm_row)))
        perm_col_mat = sp.coo_matrix((np.ones(m, dtype=mat.dtype), (perm_col, np.arange(m, dtype=perm_col.dtype))))
        return perm_row_mat.tocsr() @ mat @ perm_col_mat.tocsr()

    def reverse_cuthill_mckee(adjacency: sp.csr_matrix, node_labels: Sequence, symmetric_mode=True):
        """Reverse Cuthill McKee but also reorder the node labels and remove empty rows and columns."""
        permutation = sp.csgraph.reverse_cuthill_mckee(adjacency, symmetric_mode)
        new_adj = matrix_permute(adjacency, permutation, permutation)
        new_labels = np.array([node_labels[p] for p in permutation])
        small_n = np.max(new_adj.indices) + 1
        new_adj._shape = (small_n, small_n)
        new_adj.indptr = new_adj.indptr[:small_n + 1]
        new_adj.sort_indices()
        new_labels = new_labels[:small_n + 1]
        return new_adj, new_labels

    def graph_largest_component(adjacency: sp.csr_matrix, node_labels: Sequence):
        n_components, component_labels = sp.csgraph.connected_components(adjacency)
        component_sizes = np.bincount(component_labels)
        largest_label = np.argmax(component_sizes)
        nodes = np.where(component_labels == largest_label)[0]
        new_labels = [node_labels[n] for n in nodes]
        return adjacency[np.ix_(nodes, nodes)], new_labels

    def download_ensembl_transcripts():
        """Download the table relating genes and the proteins they transcribe from ensembl.org"""
        query_xml = \
            '<?xml version="1.0" encoding="UTF-8"?>' \
            '<!DOCTYPE Query>' \
            '<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="0" datasetConfigVersion="0.6">' \
            '<Dataset name="hsapiens_gene_ensembl" interface="default">' \
            '<Attribute name="ensembl_transcript_id"/>' \
            '<Attribute name="ensembl_gene_id"/>' \
            '<Attribute name="ensembl_peptide_id"/>' \
            '</Dataset>' \
            '</Query>'
        url = 'https://www.ensembl.org/biomart/martservice?query=' + urllib.parse.quote(query_xml)
        transcripts: set[Tuple[int, int, int]] = set()
        with urllib.request.urlopen(url) as tsv:
            for row in csv.DictReader(io.TextIOWrapper(tsv), delimiter='\t'):
                if row['Protein stable ID']:
                    transcript_ensembl_id = int(row['Transcript stable ID'][4:])
                    gene_ensembl_id = int(row['Gene stable ID'][4:])
                    protein_ensembl_id = int(row['Protein stable ID'][4:])
                    transcripts.add((transcript_ensembl_id, gene_ensembl_id, protein_ensembl_id))
        return transcripts

    def load_protein_graph(input_path):
        """The input format is specific to the STRING database."""
        mat_row, mat_col, mat_data = [], [], []
        with gzip.open(input_path, 'rt') as protein_links:
            reader = csv.reader(protein_links, delimiter=' ')
            next(reader)  # skip header "protein1 protein2 combined_score"
            for row in reader:
                protein1_ensembl_id = int(row[0][9:])
                protein2_ensembl_id = int(row[1][9:])
                combined_score = int(row[2])
                mat_row.append(protein1_ensembl_id)
                mat_col.append(protein2_ensembl_id)
                mat_data.append(combined_score)
        mat_row = np.array(mat_row, dtype=np.int32)
        mat_col = np.array(mat_col, dtype=np.int32)
        mat_data = np.array(mat_data, dtype=np.int16)
        return sp.coo_matrix((mat_data, (mat_row, mat_col)))

    def construct_gene_graph(protein_adjacency: sp.coo_matrix, gene_transcripts: Iterable[Tuple[int, int, int]]):
        """For each protein–protein link, find the two genes that transcribe them. The degree of association between
        these genes will be that of strongest link between their proteins."""
        protein2gene: dict[int, int] = dict()
        for transcript_id, gene, protein in gene_transcripts:
            protein2gene[protein] = gene
        proteins_in_string = len(np.unique(np.concatenate((protein_adjacency.row, protein_adjacency.col))))
        node2gene: list[int] = [-1] * proteins_in_string
        gene2node: dict[int, int] = dict()
        node_max = 0
        gene_adjacency = sp.lil_matrix((proteins_in_string, proteins_in_string), dtype=protein_adjacency.dtype)
        for i in tqdm.trange(len(protein_adjacency.data), desc="Processing protein links", unit=" links"):
            proteins = (protein_adjacency.row[i], protein_adjacency.col[i])
            genes = tuple(map(protein2gene.get, proteins))
            if not all(genes) or len(set(genes)) != len(genes):
                continue
            nodes = list(map(gene2node.get, genes))
            for j, node in enumerate(nodes):
                if node is None:
                    nodes[j] = node_max
                    node2gene[node_max] = genes[j]
                    gene2node[genes[j]] = node_max
                    node_max += 1
            n1, n2 = nodes
            gene_adjacency[n1, n2] = max(gene_adjacency[n1, n2], protein_adjacency.data[i])
        return gene_adjacency, node2gene

    def simplify_gene_graph(gene_adjacency, node2gene, cutoff: Optional[int] = None):
        gene_adjacency = gene_adjacency.tocsr()
        if cutoff:
            gene_adjacency.data[gene_adjacency.data < cutoff] = 0
            gene_adjacency.data[gene_adjacency.data >= cutoff] = 1
            gene_adjacency.eliminate_zeros()
            gene_adjacency.data = gene_adjacency.data.astype(np.bool_)
            gene_adjacency, node2gene = graph_largest_component(gene_adjacency, node2gene)
        gene_adjacency, node2gene = reverse_cuthill_mckee(gene_adjacency, node2gene, symmetric_mode=True)
        return gene_adjacency, node2gene

    if not quiet:
        print("Loading protein links...")
    if not os.path.exists(protein_graph_path):
        stringdb_url = 'https://stringdb-static.org/download/protein.links.v11.5/9606.protein.links.v11.5.txt.gz'
        urllib.request.urlretrieve(stringdb_url, protein_graph_path)
    stringdb = load_protein_graph(protein_graph_path)
    if not quiet:
        print("Loading gene transcripts...")
    ensembl_transcripts = download_ensembl_transcripts()
    result_adjacency, result_labels = construct_gene_graph(stringdb, ensembl_transcripts)
    result_adjacency, result_labels = simplify_gene_graph(result_adjacency, result_labels, cutoff_score)
    ensg_labels = np.array([f"ENSG{gene_ensembl_id:011}" for gene_ensembl_id in result_labels])
    save_graph(output_file, result_adjacency, node_labels=ensg_labels)


def shuffle_graph(adjacency: sp.csr_matrix, iterations: Optional[int] = None, disable_tqdm=True):
    """
    Shuffle edges in the simple undirected graph randomly while preserving degrees.
    Connected components are not preserved.
    """

    def csr_coords2index(mat: sp.csr_matrix, row: int, col: int):
        """Find idx such that mat.data[idx] == mat[row, col]. Return -1 if none exists."""
        p, q = mat.indptr[row], mat.indptr[row + 1]
        idx = mat.indices[p:q].searchsorted(col, side='left')
        return idx if idx < mat.indices.shape[0] and mat.indices[idx] == col else -1

    def csr_index2coords(mat: sp.csr_matrix, idx: int):
        """Find (row, col) such that mat[row, col] == mat.data[idx]."""
        row = mat.indptr.searchsorted(idx, side='right') - 1
        col = mat.indices[idx]
        return row, col

    def csr_replace_item(mat: sp.csr_matrix, row: int, old_col: int, new_col: int):
        """Efficiently execute mat[row, new_col] = mat[row, old_col]; mat[row, old_col] = 0"""
        p, q = mat.indptr[row], mat.indptr[row + 1]
        row_indices = mat.indices[p:q]
        idx = row_indices.searchsorted(old_col, side='left')
        row_indices[idx] = new_col
        row_indices.sort()  # Preserve mat.has_sorted_indices for future binary search operations.
        mat.indices[p:q] = row_indices

    def switch_two_edges(adj: sp.csr_matrix, e_x1y1, e_x2y2):
        """Find two edges e1: x1<->y1 and e2: x2<->y2 such that
        the four nodes are distinct and no e3: x1<->y2 or e4: x2<->y1 edge is present.
        Remove e1, e2 and insert e3, e4. Return true if successful.
        This operation preserves graph degrees."""
        x1, y1 = csr_index2coords(adj, e_x1y1)
        x2, y2 = csr_index2coords(adj, e_x2y2)
        if len({x1, y1, x2, y2}) != 4:
            return False
        if csr_coords2index(adj, x1, y2) >= 0 or csr_coords2index(adj, x2, y1) >= 0:
            return False
        csr_replace_item(adj, x1, y1, y2)
        csr_replace_item(adj, x2, y2, y1)
        csr_replace_item(adj, y1, x1, x2)
        csr_replace_item(adj, y2, x2, x1)
        return True

    adjacency.sort_indices()  # Sorted indices are required by binary search.

    # Try to do Fisher-Yates shuffles if iterations is an integer multiple of |E|-1
    n = adjacency.nnz
    if iterations is None:
        iterations = n - 1
    for k in tqdm.trange(iterations, desc="Shuffling", unit="switch", disable=disable_tqdm):
        i = k % (n - 1)
        for _ in range(2 * (n - i)):  # quit after too many tries
            j = random.randrange(i, n)
            if switch_two_edges(adjacency, i, j):
                break


def _create_permutations_subprocess(args):
    """This code is defined in a top level function so that it can be pickled during use by multiprocessing.Pool"""
    output_path, (mat, iterations) = args
    shuffled: sp.csr_matrix = mat.copy()
    shuffle_graph(shuffled, iterations)
    save_graph(output_path, shuffled)


def create_permutations(mat: sp.csr_matrix, permutation_paths: Iterable, iterations: int,
                        worker_processes: int, disable_tqdm=False):
    progress = tqdm.tqdm(permutation_paths, desc="Creating permutations",
                         unit="permutation", disable=disable_tqdm)
    constants = (mat, iterations)
    jobs = zip(progress, itertools.repeat(constants))
    if worker_processes > 1:
        with Pool(worker_processes) as pool:
            for _ in pool.imap_unordered(_create_permutations_subprocess, jobs):
                pass
    else:
        for _ in map(_create_permutations_subprocess, jobs):
            pass


NetworkPropagation = Callable[[sp.spmatrix], LinearOperator]
"""Return the similarity matrix or kernel from an adjacency matrix."""

SparseSolver = Callable[..., Tuple[np.ndarray, int]]
"""A sparse linear system solver function from scipy.sparse.linalg."""


def generalized_graph_degree_matrix(adj: sp.spmatrix) -> sp.dia_matrix:
    """Find the generalized graph degree matrix from the adjacency matrix."""
    m, n = adj.shape
    return sp.spdiags(adj.sum(axis=0, dtype=np.float64), 0, m, n)


def _lazy_sparse_inverse_matvec(y: np.ndarray, mat: sp.csc_matrix, solver: SparseSolver, scale: float):
    """This code is defined in a top level function so that it can be pickled during use by multiprocessing.Pool"""
    x, exit_code = solver(mat, y * scale)
    assert exit_code == 0, f"SparseSolver failed. {exit_code=}"
    return x


def lazy_sparse_inverse(mat: sp.spmatrix, solver: SparseSolver, scale: float = 1) -> LinearOperator:
    """Get the inverse of a sparse matrix in a lazy way:
    The needed entries of the inverse are only computed at multiplication time."""
    matvec = functools.partial(_lazy_sparse_inverse_matvec, mat=mat.tocsc(), scale=scale, solver=solver)
    # noinspection PyArgumentList
    return LinearOperator(mat.shape, matvec, dtype=np.float64)


def _rwr(transition_matrix: sp.spmatrix, alpha: float, solver: SparseSolver):
    """Random walk with restart similarity matrix from an arbitrary transition matrix."""
    system_matrix = sp.identity(transition_matrix.shape[0]) - alpha * transition_matrix
    return lazy_sparse_inverse(system_matrix, solver, 1 - alpha)


def rwr_left_similarity(adjacency: sp.spmatrix, alpha: float, solver: SparseSolver):
    """Random walk with restart similarity matrix."""
    d = generalized_graph_degree_matrix(adjacency)
    d.data[d.data != 0] = 1.0 / d.data[d.data != 0]
    transition_matrix = adjacency @ d
    return _rwr(transition_matrix, alpha, solver)


def rwr_right_similarity(adjacency: sp.spmatrix, alpha: float, solver: SparseSolver):
    """Random walk with restart similarity matrix with a right stochastic transition matrix."""
    d = generalized_graph_degree_matrix(adjacency)
    d.data[d.data != 0] = 1.0 / d.data[d.data != 0]
    transition_matrix = d @ adjacency
    return _rwr(transition_matrix, alpha, solver)


def rwr_kernel(adjacency: sp.spmatrix, alpha: float, solver: SparseSolver):
    """Random walk with restart kernel with a symmetric normalized Laplacian transition matrix"""
    d = generalized_graph_degree_matrix(adjacency)
    d.data[d.data > 0] = np.float_power(d.data[d.data > 0], -0.5)
    transition_matrix = d @ adjacency @ d
    return _rwr(transition_matrix, alpha, solver)


def rct_kernel(adjacency: sp.spmatrix, alpha: float, solver: SparseSolver):
    """Regularized commute time kernel."""
    d = generalized_graph_degree_matrix(adjacency)
    system_matrix = d - alpha * adjacency
    return lazy_sparse_inverse(system_matrix, solver)


def _lazy_sparse_exponential_matvec(y: np.ndarray, mat: sp.spmatrix) -> np.ndarray:
    """This code is defined in a top level function so that it can be pickled during use by multiprocessing.Pool"""
    return sp.linalg.expm_multiply(mat, y)


def lazy_sparse_exponential(mat: sp.spmatrix) -> LinearOperator:
    """Get the exponential of a sparse matrix in a lazy way:
    The needed entries of the exponential are only computed at multiplication time."""
    matvec = functools.partial(_lazy_sparse_exponential_matvec, mat=mat)
    # noinspection PyArgumentList
    return LinearOperator(mat.shape, matvec, dtype=np.float64)


def diffusion_kernel(adjacency: sp.spmatrix, alpha: float):
    """Exponential diffusion kernel."""
    return lazy_sparse_exponential(alpha * adjacency)


def laplacian_diffusion_kernel(adjacency: sp.spmatrix, alpha: float):
    """Laplacian exponential diffusion kernel."""
    lap = sp.csgraph.laplacian(adjacency.astype(dtype=np.float64))
    return lazy_sparse_exponential(-alpha * lap)


def _evaluate_subprocess(args):
    """This code is defined in a top level function so that it can be pickled during use by multiprocessing.Pool"""
    permutation_path, (prop, weight_vectors) = args
    permutation_path: str
    prop: NetworkPropagation
    weight_vectors: np.ndarray

    permutation_adjacency = load_graph(permutation_path)
    permutation_kernel = prop(permutation_adjacency)
    permutation_scores = np.zeros((len(weight_vectors), permutation_kernel.shape[1]))
    for i, w in enumerate(weight_vectors):
        permutation_scores[i] = permutation_kernel.matvec(w)
    return permutation_scores


def evaluate(weight_vectors: Union[np.ndarray, Sequence[np.ndarray]],
             prop: NetworkPropagation,
             adjacency: sp.spmatrix,
             permutation_paths: Optional[Sequence[str]] = None,
             pool_processes: int = max(os.cpu_count() // 2, 1),
             memory_limit: Optional[int] = 2 * 10 ** 9,
             disable_tqdm: bool = False):
    """
    Perform multiple network propagations and validate the results with permutation testing.

    :param weight_vectors: List of node weight vectors, each given to the propagation algorithm as input.
    :param prop: Network propagation algorithm.
    :param adjacency: Adjacency matrix of the network.
    :param permutation_paths: List of paths to graph permutation files.
     A graph permutation file is a shuffled version of the adjacency matrix saved with `save_graph`.
     If none are specified, permutation testing is not performed. All significance values will be 100%.
    :param pool_processes: Permutation tests are computed in parallel with `pool_workers` subprocesses.
    :param memory_limit: Do not allocate more bytes while permutation testing.
     Split the weight vectors into batches instead. This should be set to a lower number if using many worker
     processes or swap activity is too high.
    :param disable_tqdm: True to turn off tqdm progress bar.
    :return: Arrays `score_vectors` and `significance_vectors` where `score_vectors[i, j]` is the propagation score of
    the j-th node based on the i-th weight set and `significance_vectors[i, j]` is the statistical significance
     (p-value) of `score_vectors[i, j]` according to the permutation tests.
    """

    adj_kernel = prop(adjacency)
    score_vectors = np.zeros((len(weight_vectors), adj_kernel.shape[0]))
    significance_vectors = np.zeros(shape=score_vectors.shape)
    progress_eval = tqdm.tqdm(weight_vectors, desc="Evaluating",
                              unit="propagation", disable=disable_tqdm)
    for i, w in enumerate(progress_eval):
        score_vectors[i] = adj_kernel.matvec(w)
    if not permutation_paths:
        return score_vectors, significance_vectors
    pool = None
    progress = tqdm.tqdm(total=len(permutation_paths) * len(weight_vectors),
                         desc="Permutation testing", unit="propagation", disable=disable_tqdm)
    try:
        if pool_processes > 1:
            pool = Pool(pool_processes)
            map_func = pool.imap_unordered
        else:
            map_func = map
        if memory_limit is not None:
            mem_usage = pool_processes * len(weight_vectors) * adj_kernel.shape[1] * np.dtype(np.float64).itemsize
            batch_size = len(weight_vectors) // int(np.ceil(mem_usage / memory_limit))
            batches = [slice(i, min(i + batch_size, len(weight_vectors)))
                       for i in range(0, len(weight_vectors), batch_size)]
        else:
            batches = [slice(0, len(weight_vectors))]
        for batch in batches:
            constants = (prop, weight_vectors[batch])
            jobs = zip(permutation_paths, itertools.repeat(constants))
            this_batch_scores: np.ndarray = score_vectors[batch]
            for perm_scores in map_func(_evaluate_subprocess, jobs):
                significances = perm_scores >= this_batch_scores
                # When the graph is not connected, zeros can appear in perm_scores.
                # Count these nodes as significant.
                significances[perm_scores == 0] = True
                significance_vectors[batch] += significances.astype(int)
                progress.update(batch.stop - batch.start)
    finally:
        progress.close()
        if pool is not None:
            pool.close()
    significance_vectors = significance_vectors / len(permutation_paths)
    return score_vectors, significance_vectors


def node_ranking(weight_vector, score_vector, significance_vector, significance_level: float = 1):
    """Return the list of nodes by decreasing score that are not included in the starting node set of the propagation.
    Optionally also exclude nodes that were deemed significant by permutation testing."""
    ranking = [(score, sig, node) for node, weight, score, sig in
               zip(range(len(weight_vector)), weight_vector, score_vector, significance_vector)
               if weight == 0 and sig <= significance_level]
    ranking.sort(reverse=True)  # sorting by lexicographic order is faster than using a key lambda
    return [node for score, sig, node in ranking]


def main(argv):
    default_solver: SparseSolver = functools.partial(sp.linalg.lgmres, tol=1e-5, atol=1e-5)

    propagation_algos: dict[str, NetworkPropagation] = {
        rwr_left_similarity.__name__: functools.partial(rwr_left_similarity, solver=default_solver),
        rwr_right_similarity.__name__: functools.partial(rwr_right_similarity, solver=default_solver),
        rwr_kernel.__name__: functools.partial(rwr_kernel, solver=default_solver),
        rct_kernel.__name__: functools.partial(rct_kernel, solver=default_solver),
        laplacian_diffusion_kernel.__name__: laplacian_diffusion_kernel,
        diffusion_kernel.__name__: diffusion_kernel
    }

    parser = argparse.ArgumentParser(description='Perform network propagation')
    parser.add_argument('-q', '--quiet', action="store_true", help="Do not print status messages or progress bars.")
    subparsers = parser.add_subparsers(title='Subcommands', dest='subcommand')

    prp = subparsers.add_parser('propagate', help='Run the propagation.')
    prp.add_argument('network', type=argparse.FileType('rb'),
                     help=f"The network to be propagated on. Path to a file generated by `{sys.argv[0]} import-network`"
                          f" or `{sys.argv[0]} create-gene-network`.")
    prp.add_argument('--input-csv', type=argparse.FileType('rt'),
                     help="Each row is a propagation task: a set of nodes from which the propagation is to be started.")
    prp.add_argument('--output-csv', type=argparse.FileType('wt'),
                     help="For each propagation task a list of relevant associated nodes"
                          " in decreasing order of propagation score.")
    prp.add_argument('--complete-output', action='store_true',
                     help="If specified, the output row format will be: propagation task index, node, "
                          "propagation score, result significance (p-value).")
    prp.add_argument('--algo', choices=propagation_algos.keys(), default=rwr_left_similarity.__name__,
                     # rwr_left_similarity, rwr_right_similarity, rwr_kernel, rct_kernel,
                     # laplacian_diffusion_kernel, diffusion_kernel
                     help="Choice of propagation algorithm: random walk with restart (default), "
                          "RWR with right stochastic transition matrix, "
                          "RWR kernel with symmetric normalized Laplacian transition matrix, "
                          "regularized commute time kernel, "
                          "Laplacian exponential diffusion kernel, "
                          "exponential diffusion kernel.")
    prp.add_argument('--alpha', type=float, default=0.6,
                     help="The alpha parameter of the propagation algorithm (default 0.6).")
    prp.add_argument('--permutation-directory', type=pathlib.PurePath,
                     help=f"Directory containing permutations of the network. See `{sys.argv[0]} permute --help`. "
                          f"If not specified, permutation testing will not be performed and "
                          f"all significance values will be set to 1.0")
    prp.add_argument('--permutation-count', type=int,
                     help="Limit the number of permutations used.")
    prp.add_argument('--significance-threshold', type=float, default=0.05,
                     help="Real number between 0 and 1. Propagation scores with significance (p-value) "
                          "less than the threshold are excluded from the output. Default: 0.05")
    prp.add_argument('--worker-processes', type=int, default=max(os.cpu_count() // 2, 1),
                     help="The number of subprocesses to start for parallel permutation testing."
                          " Default: half of the available CPU cores.")
    prp.add_argument('--memory-limit', type=int,
                     help="The amount of memory usage of buffers allocated by many worker processes can be very high. "
                          "Set a limit in bytes.")

    pep = subparsers.add_parser('permute', help='Create permutation networks.')
    pep.add_argument('network', type=argparse.FileType('rb'),
                     help=f"The network to be permuted. Path to a file generated by `{sys.argv[0]} import-network`"
                          f" or `{sys.argv[0]} create-gene-network`.")
    pep.add_argument('--output-directory', type=pathlib.PurePath, required=True,
                     help="The created files will follow the naming scheme OUTPUT_DIRECTORY/permutation_*.npz")
    pep.add_argument('--count', type=int, required=True,
                     help="Number of permutations to create.")
    pep.add_argument('--iterations', type=float, default=1,
                     help="Real number. The number of edge switches performed for each permutation will be "
                          "the number of edges of the network times this number.")
    pep.add_argument('--worker-processes', type=int, default=max(os.cpu_count() // 2, 1),
                     help="The number of subprocesses to start for parallel permuting."
                          " Default: half of the available CPU cores.")

    inp = subparsers.add_parser('import-network', help="Import a network file from csv.")
    inp.add_argument('output-network', type=argparse.FileType('wb'),
                     help="Output file path with .npz extension.")
    inp.add_argument('--input-csv', type=argparse.FileType('rt'), required=True,
                     help="The format is two adjacent nodes for each row.")

    enp = subparsers.add_parser('export-network', help='Export a network file to csv.')
    enp.add_argument('network', type=argparse.FileType('rb'),
                     help=f"The network to be permuted. Path to a file generated by `{sys.argv[0]} import-network`"
                          f" or `{sys.argv[0]} create-gene-network`.")
    enp.add_argument('--output-csv', type=argparse.FileType('wt'), required=True,
                     help="The format is two adjacent nodes for each row.")
    enp.add_argument('--with-header', action='store_true',
                     help="Include a header with all nodes.")

    cgn = subparsers.add_parser('create-gene-network', help='Create the gene association network.')
    cgn.add_argument('output-network', type=argparse.FileType('wb'),
                     help="Output file path with .npz extension.")
    cgn.add_argument('--cutoff-score', type=int, default=700,
                     help="Protein-protein interaction entries with score lower than the cutoff are disregarded. "
                          "Default: 700.")
    cgn.add_argument('--protein-links-path', type=pathlib.PurePath,
                     default='9606.protein.links.v11.5.txt.gz',
                     help="Path to the STRING (https://string-db.org) protein-protein interaction network. "
                          "If none is provided 9606.protein.links.v11.5.txt.gz will be downloaded.")

    args = vars(parser.parse_args(argv))

    # if args.get('output_csv') and args['output_csv'].name == '<stdout>':
    #    args['quiet'] = True

    quiet = args.get('quiet')
    if quiet:
        # noinspection PyUnusedLocal, PyShadowingNames
        def noop(it, *a, **k):
            return it

        # noinspection PyUnusedLocal, PyShadowingNames
        def trange(stop, *a, **k):
            return range(stop)

        tqdm.tqdm = noop
        tqdm.trange = trange

    if args['subcommand'] == 'propagate':
        worker_processes = args.get('worker_processes')
        mem_lim = args.get('memory_limit')
        prop: NetworkPropagation = functools.partial(propagation_algos[args['algo']], alpha=args['alpha'])
        permutations = None
        if args.get('permutation_directory'):
            permutations = glob.glob(os.path.join(args['permutation_directory'], 'permutation_*.npz'))
            permutations.sort()
            if args.get('permutation_count'):
                permutations = permutations[:args['permutation_count']]
        adjacency, node2label, label2node = load_graph(args['network'], labels=True)

        weight_vectors = []
        for row in csv.reader(args['input_csv']):
            weight = np.zeros(adjacency.shape[0])
            for label in row:
                try:
                    weight[label2node[label]] = 1
                except KeyError as k:
                    raise KeyError("The input CSV contains a label that is outside of the network.") from k
            weight_vectors.append(weight)
        score_vectors, significance_vectors = evaluate(weight_vectors=weight_vectors, prop=prop, adjacency=adjacency,
                                                       permutation_paths=permutations,
                                                       pool_processes=worker_processes, memory_limit=mem_lim,
                                                       disable_tqdm=quiet)
        writer = csv.writer(args['output_csv'])
        if not args.get('complete_output'):
            for weight, sc, sig in zip(weight_vectors, score_vectors, significance_vectors):
                ranking = node_ranking(weight, sc, sig, args['significance_threshold'])
                writer.writerow(node2label[node] for node in ranking)
        else:
            for weight_vector_index in range(len(weight_vectors)):
                for node, score, sig in sorted(zip(range(len(weight_vectors[weight_vector_index])),
                                                   score_vectors[weight_vector_index],
                                                   significance_vectors[weight_vector_index]),
                                               key=lambda x: x[1], reverse=True):
                    writer.writerow((weight_vector_index, node2label[node], score, sig))
        exit()

    if args['subcommand'] == 'permute':
        worker_processes = args.get('worker_processes')
        adjacency = load_graph(args['network'], labels=False)
        count = args['count']
        files = [os.path.join(args['output_directory'], f"permutation_{n:0{int(np.ceil(np.log10(count)))}}.npz")
                 for n in range(count)]
        create_permutations(mat=adjacency, permutation_paths=files,
                            iterations=int((adjacency.nnz - 1) * args['iterations']),
                            worker_processes=worker_processes,
                            disable_tqdm=quiet)
        exit()

    if args['subcommand'] == 'import-network':
        label2node = dict()
        node2label = []
        edges = []
        weighted = False
        for row in csv.reader(args['input_csv']):
            w = True
            if len(row) == 2:
                label_a, label_b = row
            elif len(row) == 3:
                weighted = True
                label_a, label_b, w = row
                w = float(w)
            else:
                raise KeyError("Input CSV contains invalid columns.")
            node_a = label2node.get(label_a)
            if node_a is None:
                label2node[label_a] = len(node2label)
                node2label.append(label_a)
            node_b = label2node.get(label_b)
            if node_b is None:
                label2node[label_b] = len(node2label)
                node2label.append(label_b)
            edges.append((node_a, node_b, w))
        dtype = float if weighted else bool
        adjacency = sp.lil_matrix((len(node2label), len(node2label)), dtype=dtype)
        for node_a, node_b, w in edges:
            adjacency[node_a, node_b] = w
            adjacency[node_b, node_a] = w
        save_graph(args['output-network'], adjacency, node_labels=node2label)
        exit()

    if args['subcommand'] == 'export-network':
        adjacency, node2label, label2node = load_graph(args['network'], labels=True)
        writer = csv.writer(args['output_csv'])
        if args.get('with_header'):
            writer.writerow(node2label)
        adjacency_coo: sp.coo_matrix = adjacency.tocoo()
        weighted = adjacency.dtype != bool
        if weighted:
            for i, node_a, node_b in zip(range(adjacency_coo.nnz), adjacency_coo.row, adjacency_coo.col):
                writer.writerow((node2label[node_a], node2label[node_b], adjacency_coo.data[i]))
        else:
            for node_a, node_b in zip(adjacency_coo.row, adjacency_coo.col):
                writer.writerow((node2label[node_a], node2label[node_b]))
        exit()

    if args['subcommand'] == 'create-gene-network':
        create_gene_graph(output_file=args['output-network'], quiet=quiet,
                          protein_graph_path=args.get('protein_links_path'),
                          cutoff_score=args.get('cutoff_score'))
        exit()

    parser.print_usage()
    exit(2)


if __name__ == '__main__':
    main(sys.argv[1:])
