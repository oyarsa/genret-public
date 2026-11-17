"""Build pairwise facet similarity matrices across all papers in the dataset.

For each of the 4 facet types (problems, methods, results, contributions),
computes similarity metrics between every pair of papers:

- `mean_pool`: Average facet items into one vector, then cosine similarity.
  Always computed; used by downstream candidate generation.
- `max_pairs`: Maximum cosine similarity over all item pairs (a∈i, b∈j).
  Optional; expensive to compute and not currently used downstream.
- `mean_pairs`: Mean cosine similarity over all valid item pairs.
  Optional; expensive to compute and not currently used downstream.

Output: one `.npz` file per facet type, each containing `(N, N)` float16
matrices and the `corpusids` array.
"""

# pyright: basic

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from genret.paper_parsing import Facets
from genret.processing.facet_evaluation import FACET_NAMES
from genret.processing.nptypes import Float16Array2D, Float32Array2D, Float32Array3D
from genret.utils import iter_jsonl, resolve_jsonl, track_progress

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class FacetArrays:
    """Pre-loaded, padded embedding arrays for one facet type across all papers.

    Args:
        embeddings: Shape `(N, max_k, D)` padded float32 array.
            Rows beyond `counts[i]` are zero-padded.
        counts: Shape `(N,)` int32 array of true item counts per paper.
    """

    embeddings: Float32Array3D
    counts: npt.NDArray[np.int32]


def _slice_facet_items(
    row: Float32Array2D, facet_counts: dict[str, int], facet_name: str
) -> Float32Array2D:
    """Extract the embedding rows for one facet type from a paper's row.

    Items in the NPZ row are laid out contiguously in `FACET_NAMES` order:
    `problems[0..p], methods[0..m], results[0..n], contributions[0..c]`.

    Args:
        row: Shape `(total_items, D)` embedding matrix for one paper.
        facet_counts: Number of items per facet type for this paper.
        facet_name: The facet type to extract (e.g. `"problems"`).

    Returns:
        Shape `(k, D)` slice, or an empty `(0, D)` array if `k == 0`.
    """
    offset = 0

    for name in FACET_NAMES:
        k = facet_counts[name]
        if name == facet_name:
            return row[offset : offset + k]

        offset += k

    return row[0:0]


def load_facet_arrays(
    facets_path: Path, embeddings_path: Path
) -> tuple[list[int], dict[str, FacetArrays]]:
    """Load facet records and embeddings, building padded arrays per facet type.

    Args:
        facets_path: Path to the facets JSONL file (plain or compressed).
        embeddings_path: Path to the `.npz` facet embeddings cache.

    Returns:
        A tuple of:
        - `corpusids`: Ordered list of corpus IDs matching row indices.
        - `arrays_by_facet`: Map from facet name to `FacetArrays`.
    """
    resolved = resolve_jsonl(facets_path)

    corpusids: list[int] = []
    item_counts: dict[str, list[int]] = {name: [] for name in FACET_NAMES}

    for record in track_progress(iter_jsonl(resolved), desc="Loading facets..."):
        facets = Facets.model_validate(record)
        corpusids.append(facets.corpusid)
        item_counts["problems"].append(len(facets.problems))
        item_counts["methods"].append(len(facets.methods))
        item_counts["results"].append(len(facets.results))
        item_counts["contributions"].append(len(facets.contributions))

    if not corpusids:
        logger.warning("No facet records found; returning empty arrays")
        arrays_by_facet = {
            name: FacetArrays(
                embeddings=np.zeros((0, 1, 1536), dtype=np.float32),
                counts=np.zeros(0, dtype=np.int32),
            )
            for name in FACET_NAMES
        }
        return corpusids, arrays_by_facet

    n_records = len(corpusids)
    logger.info(f"Loaded {n_records} facet records")
    logger.info(f"Loading embeddings from {embeddings_path}")

    with np.load(embeddings_path) as npz:
        embeddings_data: dict[str, Float32Array2D] = {
            key: npz[key] for key in npz.files
        }

    # Infer embedding dimensionality from first available entry
    first_row = next(iter(embeddings_data.values()), None)
    emb_dim = first_row.shape[-1] if first_row is not None else 1536

    arrays_by_facet: dict[str, FacetArrays] = {}
    for facet_name in FACET_NAMES:
        counts_list = item_counts[facet_name]
        max_k = max(counts_list) if counts_list else 0
        max_k = max(max_k, 1)  # avoid zero-dim arrays

        padded = np.zeros((n_records, max_k, emb_dim), dtype=np.float32)
        counts_arr = np.array(counts_list, dtype=np.int32)

        for i, cid in enumerate(corpusids):
            key = str(cid)
            if key not in embeddings_data:
                counts_arr[i] = 0
                continue

            row = embeddings_data[key]
            facet_counts_for_row = {name: item_counts[name][i] for name in FACET_NAMES}
            facet_slice = _slice_facet_items(row, facet_counts_for_row, facet_name)

            k = facet_slice.shape[0]
            if k > 0:
                padded[i, :k, :] = facet_slice
            counts_arr[i] = k

        arrays_by_facet[facet_name] = FacetArrays(embeddings=padded, counts=counts_arr)
        missing = int((counts_arr == 0).sum())
        logger.info(f"  {facet_name}: max_k={max_k}, missing={missing}/{n_records}")

    return corpusids, arrays_by_facet


def mean_pool_vectors(arrays: FacetArrays) -> Float32Array2D:
    """Compute L2-normalised mean-pooled vectors for each paper.

    Each paper facet is represented by the mean of its item embeddings,
    then L2-normalised.  Papers with no items (count=0) get a zero vector.

    Args:
        arrays: Pre-loaded padded embedding arrays for one facet type.

    Returns:
        Shape `(N, D)` float32 normalised vectors.
    """
    vecs: Float32Array2D = arrays.embeddings.sum(axis=1)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-12)


def compute_mean_pool(arrays: FacetArrays) -> Float16Array2D:
    """Compute mean-pooled cosine similarity matrix.

    Each paper facet is represented by the mean of its (L2-normalised) item
    embeddings, then pairwise cosine similarities are computed.  Papers with
    no items (count=0) receive a zero vector and produce zero similarities.

    Args:
        arrays: Pre-loaded padded embedding arrays for one facet type.

    Returns:
        Shape `(N, N)` float16 cosine similarity matrix.
    """
    normed_vecs = mean_pool_vectors(arrays)
    result: Float32Array2D = normed_vecs @ normed_vecs.T
    return result.astype(np.float16)


def compute_cross_mean_pool(
    query_arrays: FacetArrays,
    key_arrays: FacetArrays,
) -> Float16Array2D:
    """Compute cross-facet mean-pooled cosine similarity matrix.

    Compares query papers' facet embeddings against key papers' facet
    embeddings from a *different* facet type. For example, paper A's
    problems against paper B's contributions.

    Args:
        query_arrays: Embeddings for the query facet type (e.g. problems).
        key_arrays: Embeddings for the key facet type (e.g. contributions).

    Returns:
        Shape `(N, N)` float16 cosine similarity matrix where entry
        `[i, j]` is the similarity between paper i's query facet and
        paper j's key facet.
    """
    query_vecs: Float32Array2D = query_arrays.embeddings.sum(axis=1)
    key_vecs: Float32Array2D = key_arrays.embeddings.sum(axis=1)

    q_norms = np.linalg.norm(query_vecs, axis=1, keepdims=True)
    k_norms = np.linalg.norm(key_vecs, axis=1, keepdims=True)

    normed_q = query_vecs / np.maximum(q_norms, 1e-12)
    normed_k = key_vecs / np.maximum(k_norms, 1e-12)

    result: Float32Array2D = normed_q @ normed_k.T
    return result.astype(np.float16)


def _compute_cross_pairwise_gpu(
    query_arrays: FacetArrays,
    key_arrays: FacetArrays,
    block_size: int,
    *,
    progress_desc: str = "Cross pairwise (GPU)...",
) -> tuple[Float16Array2D, Float16Array2D]:
    """Compute cross-facet max-pairs and mean-pairs on GPU.

    Unlike same-facet pairwise, the matrix is *not* symmetric because
    query and key come from different facet types.

    Args:
        query_arrays: Embeddings for the query facet type.
        key_arrays: Embeddings for the key facet type.
        block_size: Number of papers processed per block.
        progress_desc: Description for the progress bar.

    Returns:
        A tuple ``(max_pairs, mean_pairs)``, each ``(N, N)`` float16.
    """
    import torch  # noqa: PLC0415

    n_q, max_k_q, d = query_arrays.embeddings.shape
    n_k, max_k_k, _ = key_arrays.embeddings.shape
    assert n_q == n_k, "Query and key must have the same number of papers"
    n = n_q
    device = torch.device("cuda")

    q_flat = torch.from_numpy(
        np.ascontiguousarray(query_arrays.embeddings).reshape(-1, d)
    ).to(device=device, dtype=torch.float16)
    k_flat = torch.from_numpy(
        np.ascontiguousarray(key_arrays.embeddings).reshape(-1, d)
    ).to(device=device, dtype=torch.float16)

    q_valid = torch.from_numpy(
        np.arange(max_k_q)[None, :] < query_arrays.counts[:, None]
    ).to(device)
    k_valid = torch.from_numpy(
        np.arange(max_k_k)[None, :] < key_arrays.counts[:, None]
    ).to(device)

    # Mean vectors
    q_counts = (
        torch.from_numpy(np.maximum(query_arrays.counts, 1).astype(np.float32))
        .to(device)
        .unsqueeze(1)
    )
    k_counts = (
        torch.from_numpy(np.maximum(key_arrays.counts, 1).astype(np.float32))
        .to(device)
        .unsqueeze(1)
    )
    q_emb = q_flat.reshape(n, max_k_q, d).float()
    k_emb = k_flat.reshape(n, max_k_k, d).float()
    q_mean = ((q_emb * q_valid.unsqueeze(2)).sum(dim=1) / q_counts).half()
    k_mean = ((k_emb * k_valid.unsqueeze(2)).sum(dim=1) / k_counts).half()

    max_out = np.zeros((n, n), dtype=np.float16)
    mean_out = np.zeros((n, n), dtype=np.float16)

    for bs in track_progress(range(0, n, block_size), desc=progress_desc):
        be = min(bs + block_size, n)
        b_q = be - bs

        qf = q_flat[bs * max_k_q : be * max_k_q]
        qv = q_valid[bs:be]
        qm = q_mean[bs:be]

        for ks in range(0, n, block_size):
            ke = min(ks + block_size, n)
            b_k = ke - ks

            # Mean pairs
            block_mean = (qm @ k_mean[ks:ke].T).cpu().numpy()
            mean_out[bs:be, ks:ke] = block_mean

            # Max pairs
            kf = k_flat[ks * max_k_k : ke * max_k_k]
            sim = (qf @ kf.T).reshape(b_q, max_k_q, b_k, max_k_k)

            kv = k_valid[ks:ke]
            sim = sim.masked_fill(~kv[None, None, :, :], float("-inf"))
            sim_max_k = sim.amax(dim=3)

            sim_max_k = sim_max_k.masked_fill(~qv[:, :, None], float("-inf"))
            block_max = sim_max_k.amax(dim=1)

            block_max = block_max.clamp(min=0.0).cpu().numpy()
            max_out[bs:be, ks:ke] = block_max

    return max_out, mean_out


def _compute_cross_pairwise_cpu(
    query_arrays: FacetArrays,
    key_arrays: FacetArrays,
    block_size: int,
    *,
    progress_desc: str = "Cross pairwise (CPU)...",
) -> tuple[Float16Array2D, Float16Array2D]:
    """Compute cross-facet max-pairs and mean-pairs on CPU.

    Args:
        query_arrays: Embeddings for the query facet type.
        key_arrays: Embeddings for the key facet type.
        block_size: Number of papers processed per block.
        progress_desc: Description for the progress bar.

    Returns:
        A tuple ``(max_pairs, mean_pairs)``, each ``(N, N)`` float16.
    """
    n_q, max_k_q, d = query_arrays.embeddings.shape
    n_k, max_k_k, _ = key_arrays.embeddings.shape
    assert n_q == n_k, "Query and key must have the same number of papers"
    n = n_q

    q_valid = np.arange(max_k_q)[None, :] < query_arrays.counts[:, None]
    k_valid = np.arange(max_k_k)[None, :] < key_arrays.counts[:, None]

    q_counts_safe = np.maximum(query_arrays.counts, 1).astype(np.float32)[:, None]
    k_counts_safe = np.maximum(key_arrays.counts, 1).astype(np.float32)[:, None]
    q_masked = np.where(q_valid[:, :, None], query_arrays.embeddings, 0.0)
    k_masked = np.where(k_valid[:, :, None], key_arrays.embeddings, 0.0)
    q_mean = q_masked.sum(axis=1) / q_counts_safe
    k_mean = k_masked.sum(axis=1) / k_counts_safe

    q_flat = np.ascontiguousarray(query_arrays.embeddings).reshape(-1, d)
    k_flat_all = np.ascontiguousarray(key_arrays.embeddings).reshape(-1, d)

    max_out = np.zeros((n, n), dtype=np.float16)
    mean_out = np.zeros((n, n), dtype=np.float16)

    for bs in track_progress(range(0, n, block_size), desc=progress_desc):
        be = min(bs + block_size, n)
        b_q = be - bs

        qf = q_flat[bs * max_k_q : be * max_k_q]
        qv = q_valid[bs:be, :, None]  # (b_q, max_k_q, 1)
        qm = q_mean[bs:be]

        for ks in range(0, n, block_size):
            ke = min(ks + block_size, n)
            b_k = ke - ks

            block_mean = (qm @ k_mean[ks:ke].T).astype(np.float16)
            mean_out[bs:be, ks:ke] = block_mean

            kf = k_flat_all[ks * max_k_k : ke * max_k_k]
            sim = (qf @ kf.T).reshape(b_q, max_k_q, b_k, max_k_k)

            k_mask_right = k_valid[ks:ke][None, None, :, :]
            sim_max_k = sim.max(axis=3, initial=-np.inf, where=k_mask_right)

            assert qv.shape[1] == sim_max_k.shape[1] == max_k_q
            block_max = sim_max_k.max(axis=1, initial=-np.inf, where=qv)

            block_max_f16 = np.where(np.isfinite(block_max), block_max, 0.0).astype(
                np.float16
            )
            max_out[bs:be, ks:ke] = block_max_f16

    return max_out, mean_out


def compute_cross_pairwise(
    query_arrays: FacetArrays,
    key_arrays: FacetArrays,
    block_size: int,
    *,
    progress_desc: str = "Cross pairwise...",
) -> tuple[Float16Array2D, Float16Array2D]:
    """Compute cross-facet max-pairs and mean-pairs similarity matrices.

    Auto-dispatches to GPU or CPU. Unlike same-facet pairwise, the
    result matrix is not symmetric.

    Args:
        query_arrays: Embeddings for the query facet type.
        key_arrays: Embeddings for the key facet type.
        block_size: Number of papers processed per block.
        progress_desc: Description for the progress bar.

    Returns:
        A tuple ``(max_pairs, mean_pairs)``, each ``(N, N)`` float16.
    """
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            logger.info("CUDA available — using GPU acceleration with float16")
            gpu_block = max(block_size, 1024)
            return _compute_cross_pairwise_gpu(
                query_arrays, key_arrays, gpu_block, progress_desc=progress_desc
            )
    except ImportError:
        pass

    logger.info("No CUDA — falling back to CPU")
    return _compute_cross_pairwise_cpu(
        query_arrays, key_arrays, block_size, progress_desc=progress_desc
    )


def _compute_pairwise_gpu(
    arrays: FacetArrays,
    block_size: int,
    *,
    progress_desc: str = "Pairwise (GPU)...",
) -> tuple[Float16Array2D, Float16Array2D]:
    """Compute max-pairs and mean-pairs similarity matrices on GPU.

    Uses PyTorch with float16 tensor core acceleration. The full item
    embedding matrix is kept on GPU; blocks are processed in the upper
    triangle and mirrored.

    Args:
        arrays: Pre-loaded padded embedding arrays for one facet type.
        block_size: Number of query papers processed per block.
        progress_desc: Description for the progress bar.

    Returns:
        A tuple ``(max_pairs, mean_pairs)``, each an ``(N, N)`` float16
        numpy matrix.
    """
    import torch  # noqa: PLC0415

    n, max_k, d = arrays.embeddings.shape
    device = torch.device("cuda")

    # Upload everything to GPU as float16 for tensor core speedup
    flat = torch.from_numpy(np.ascontiguousarray(arrays.embeddings).reshape(-1, d)).to(
        device=device, dtype=torch.float16
    )

    valid_all = torch.from_numpy(np.arange(max_k)[None, :] < arrays.counts[:, None]).to(
        device
    )

    # Mean vectors: compute on GPU in float32 for precision, keep float16
    counts_safe = (
        torch.from_numpy(np.maximum(arrays.counts, 1).astype(np.float32))
        .to(device)
        .unsqueeze(1)
    )
    emb_gpu = flat.reshape(n, max_k, d).float()
    mask_3d = valid_all.unsqueeze(2)  # (n, max_k, 1)
    mean_vecs = (emb_gpu * mask_3d).sum(dim=1) / counts_safe  # (n, d)
    mean_vecs = mean_vecs.half()

    # Pre-allocate outputs on CPU (too large for GPU at N>30k)
    max_out = np.zeros((n, n), dtype=np.float16)
    mean_out = np.zeros((n, n), dtype=np.float16)

    for block_start in track_progress(range(0, n, block_size), desc=progress_desc):
        block_end = min(block_start + block_size, n)
        b_q = block_end - block_start

        q_flat = flat[block_start * max_k : block_end * max_k]  # (b_q*K, d)
        q_valid = valid_all[block_start:block_end]  # (b_q, K)
        q_means = mean_vecs[block_start:block_end]  # (b_q, d)

        for key_start in range(block_start, n, block_size):
            key_end = min(key_start + block_size, n)
            b_k = key_end - key_start

            # --- Mean Pairs ---
            k_means = mean_vecs[key_start:key_end]
            block_mean = (q_means @ k_means.T).cpu().numpy()

            mean_out[block_start:block_end, key_start:key_end] = block_mean
            mean_out[key_start:key_end, block_start:block_end] = block_mean.T

            # --- Max Pairs ---
            k_flat = flat[key_start * max_k : key_end * max_k]

            # (b_q*K, d) @ (d, b_k*K) -> reshape to (b_q, K, b_k, K)
            sim = (q_flat @ k_flat.T).reshape(b_q, max_k, b_k, max_k)

            # Mask invalid items with -inf, then reduce
            k_mask = valid_all[key_start:key_end]  # (b_k, K)
            sim = sim.masked_fill(~k_mask[None, None, :, :], float("-inf"))
            sim_max_k = sim.amax(dim=3)  # (b_q, K, b_k)

            sim_max_k = sim_max_k.masked_fill(~q_valid[:, :, None], float("-inf"))
            block_max = sim_max_k.amax(dim=1)  # (b_q, b_k)

            # -inf → 0 for papers with no items
            block_max = block_max.clamp(min=0.0).cpu().numpy()

            max_out[block_start:block_end, key_start:key_end] = block_max
            max_out[key_start:key_end, block_start:block_end] = block_max.T

    return max_out, mean_out


def _compute_pairwise_cpu(
    arrays: FacetArrays,
    block_size: int,
    *,
    progress_desc: str = "Pairwise (CPU)...",
) -> tuple[Float16Array2D, Float16Array2D]:
    """Compute max-pairs and mean-pairs similarity matrices on CPU.

    Uses tiled blocks to avoid materialising the full ``(N*K, N*K)``
    similarity tensor. Memory per block is approximately
    ``block_size * K * N * K * 4`` bytes.

    Args:
        arrays: Pre-loaded padded embedding arrays for one facet type.
        block_size: Number of query papers processed per block.
        progress_desc: Description for the progress bar.

    Returns:
        A tuple ``(max_pairs, mean_pairs)``, each an ``(N, N)`` float16
        numpy matrix.
    """
    n: int = arrays.embeddings.shape[0]
    max_k: int = arrays.embeddings.shape[1]
    d: int = arrays.embeddings.shape[2]

    # 0. Pre-compute valid mask for all items. shape: (n, max_k)
    valid_all = np.arange(max_k)[None, :] < arrays.counts[:, None]

    # 1. Pre-compute mean vectors safely.
    counts_safe = np.maximum(arrays.counts, 1).astype(np.float32)[:, None]
    masked_embeddings = np.where(valid_all[:, :, None], arrays.embeddings, 0.0)
    mean_vecs = masked_embeddings.sum(axis=1) / counts_safe

    # 2. Pre-allocate float16 outputs.
    max_out = np.zeros((n, n), dtype=np.float16)
    mean_out = np.zeros((n, n), dtype=np.float16)

    # 3. Setup for Max Pairs
    flat = np.ascontiguousarray(arrays.embeddings).reshape(-1, d)

    # 4. 2D Block Processing
    for block_start in track_progress(range(0, n, block_size), desc=progress_desc):
        block_end = min(block_start + block_size, n)
        b_q = block_end - block_start

        q_means = mean_vecs[block_start:block_end]
        q_flat = flat[block_start * max_k : block_end * max_k]
        q_mask = valid_all[block_start:block_end, :, None]  # (b_q, max_k, 1)

        # Process keys in blocks
        for key_start in range(block_start, n, block_size):
            key_end = min(key_start + block_size, n)
            b_k = key_end - key_start

            # --- Triangular Mean Pairs ---
            k_means = mean_vecs[key_start:key_end]
            block_mean = (q_means @ k_means.T).astype(np.float16)

            mean_out[block_start:block_end, key_start:key_end] = block_mean
            mean_out[key_start:key_end, block_start:block_end] = block_mean.T

            # --- Triangular Max Pairs ---
            k_flat = flat[key_start * max_k : key_end * max_k]

            # sim shape: (b_q, max_k, b_k, max_k)
            sim = (q_flat @ k_flat.T).reshape(b_q, max_k, b_k, max_k)

            # Reduction 1: Max over keys (axis 3)
            k_mask_right = valid_all[key_start:key_end][None, None, :, :]
            sim_max_k = sim.max(axis=3, initial=-np.inf, where=k_mask_right)

            # Reduction 2: Max over queries (axis 1)
            assert q_mask.shape[1] == sim_max_k.shape[1] == max_k
            block_max = sim_max_k.max(axis=1, initial=-np.inf, where=q_mask)

            block_max_f16 = np.where(np.isfinite(block_max), block_max, 0.0).astype(
                np.float16
            )

            max_out[block_start:block_end, key_start:key_end] = block_max_f16
            max_out[key_start:key_end, block_start:block_end] = block_max_f16.T

    return max_out, mean_out


def compute_pairwise(
    arrays: FacetArrays,
    block_size: int,
    *,
    progress_desc: str = "Pairwise...",
) -> tuple[Float16Array2D, Float16Array2D]:
    """Compute max-pairs and mean-pairs similarity matrices using tiled blocks.

    Automatically uses GPU (float16 tensor cores) when CUDA is available,
    falling back to CPU numpy. GPU block_size defaults are larger (1024)
    since GPU memory allows it.

    Args:
        arrays: Pre-loaded padded embedding arrays for one facet type.
        block_size: Number of query papers processed per block.
        progress_desc: Description for the progress bar.

    Returns:
        A tuple ``(max_pairs, mean_pairs)``, each an ``(N, N)`` float16
        matrix. For paper pairs where either paper has no items,
        similarity is 0.0.
    """
    if arrays.embeddings.shape[0] != arrays.counts.shape[0]:
        raise ValueError("Embeddings and counts must have the same number of rows")
    if block_size <= 0:
        raise ValueError("Block size must be positive")

    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            logger.info("CUDA available — using GPU acceleration with float16")
            gpu_block = max(block_size, 1024)
            return _compute_pairwise_gpu(arrays, gpu_block, progress_desc=progress_desc)
    except ImportError:
        pass

    logger.info("No CUDA — falling back to CPU")
    return _compute_pairwise_cpu(arrays, block_size, progress_desc=progress_desc)


def build_facet_graph(
    facets_path: Path,
    embeddings_path: Path,
    output_dir: Path,
    block_size: int = 256,
    *,
    include_pairwise: bool = False,
    cross_facet_pairs: list[tuple[str, str]] | None = None,
) -> None:
    """Build pairwise facet similarity matrices and write one `.npz` per facet type.

    For each of the 4 facet types, writes a file
    `facet_graph_{name}.npz` containing:

    - `corpusids`: `(N,)` int64 array.
    - `mean_pool`: `(N, N)` float16 cosine similarity (mean-pooled vectors).

    When `include_pairwise` is True, also computes and stores:

    - `max_pairs`: `(N, N)` float16 max item-pair cosine similarity.
    - `mean_pairs`: `(N, N)` float16 mean item-pair cosine similarity.

    These are expensive to compute and not used by the downstream candidate
    generation pipeline, so they are skipped by default.

    When `cross_facet_pairs` is provided, also computes cross-facet
    similarity matrices (e.g. problems vs contributions) and writes them
    as `facet_graph_{query}_x_{key}.npz`.

    Files are written via a `.tmp` intermediate to avoid partial outputs.

    Args:
        facets_path: Path to the facets JSONL file.
        embeddings_path: Path to the `.npz` facet embeddings cache.
        output_dir: Directory to write output files.
        block_size: Number of query papers per block in pairwise computation.
            Reduce if memory is tight (256 → ~700 MB/block for N=27k, K=5).
        include_pairwise: If True, compute `max_pairs` and `mean_pairs`
            matrices in addition to `mean_pool`. These are expensive and
            not used by candidate generation. Default False.
        cross_facet_pairs: Optional list of `(query_facet, key_facet)` tuples
            for cross-facet similarity matrices.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    corpusids, arrays_by_facet = load_facet_arrays(facets_path, embeddings_path)
    if not corpusids:
        logger.warning("No papers loaded; nothing to do.")
        return

    corpusids_arr = np.array(corpusids, dtype=np.int64)

    for facet_name in FACET_NAMES:
        arrays = arrays_by_facet[facet_name]
        out_path = output_dir / f"facet_graph_{facet_name}.npz"
        tmp_path = output_dir / f"facet_graph_{facet_name}.tmp.npz"

        if out_path.exists():
            logger.info(f"Skipping {facet_name} (already exists: {out_path})")
            continue

        logger.info(f"=== {facet_name} ===")

        logger.info("  Computing mean_pool...")
        corpus_vectors = mean_pool_vectors(arrays)
        mean_pool: Float16Array2D = (corpus_vectors @ corpus_vectors.T).astype(
            np.float16
        )

        if include_pairwise:
            logger.info("  Computing max_pairs and mean_pairs...")
            max_pairs, mean_pairs = compute_pairwise(
                arrays,
                block_size,
                progress_desc=f"{facet_name} pairwise",
            )
            logger.info(f"  Writing {out_path}...")
            np.savez_compressed(
                tmp_path,
                corpusids=corpusids_arr,
                mean_pool=mean_pool,
                corpus_vectors=corpus_vectors.astype(np.float16),
                max_pairs=max_pairs,
                mean_pairs=mean_pairs,
            )
        else:
            logger.info("  Skipping max_pairs/mean_pairs (include_pairwise=False)")
            logger.info(f"  Writing {out_path}...")
            np.savez_compressed(
                tmp_path,
                corpusids=corpusids_arr,
                mean_pool=mean_pool,
                corpus_vectors=corpus_vectors.astype(np.float16),
            )
        tmp_path.rename(out_path)
        logger.info(f"  Saved {out_path}")

    for query_facet, key_facet in cross_facet_pairs or []:
        cross_name = f"{query_facet}_x_{key_facet}"
        out_path = output_dir / f"facet_graph_{cross_name}.npz"
        tmp_path = output_dir / f"facet_graph_{cross_name}.tmp.npz"

        if out_path.exists():
            logger.info(f"Skipping {cross_name} (already exists: {out_path})")
            continue

        logger.info(f"=== cross-facet: {query_facet} → {key_facet} ===")
        logger.info("  Computing cross mean_pool...")
        cross_pool = compute_cross_mean_pool(
            arrays_by_facet[query_facet],
            arrays_by_facet[key_facet],
        )

        if include_pairwise:
            logger.info("  Computing cross max_pairs and mean_pairs...")
            cross_max, cross_mean = compute_cross_pairwise(
                arrays_by_facet[query_facet],
                arrays_by_facet[key_facet],
                block_size,
                progress_desc=f"{cross_name} pairwise",
            )
            logger.info(f"  Writing {out_path}...")
            np.savez_compressed(
                tmp_path,
                corpusids=corpusids_arr,
                mean_pool=cross_pool,
                max_pairs=cross_max,
                mean_pairs=cross_mean,
            )
        else:
            logger.info(f"  Writing {out_path}...")
            np.savez_compressed(
                tmp_path,
                corpusids=corpusids_arr,
                mean_pool=cross_pool,
            )
        tmp_path.rename(out_path)
        logger.info(f"  Saved {out_path}")
