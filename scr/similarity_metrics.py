import numpy as np
from scipy.stats import spearmanr, pearsonr
from skimage.metrics import structural_similarity
from skimage.feature import hog


def _select_bands(arr: np.ndarray, bands=None) -> np.ndarray:
    """
    Helper function to sub-select specific bands (channels) from the last dimension
    of a 3D array. If `arr` is 2D and bands is not None, an error is raised.

    Parameters:
    -----------
    arr  : np.ndarray
        Input array of shape (H, W) or (H, W, C).
    bands: list or None
        Indices of the channels/bands to select. If None, use all channels.

    Returns:
    --------
    np.ndarray
        The sub-selected array if 3D; unchanged if 2D and bands=None.
    """
    if bands is not None:
        if arr.ndim == 3:
            arr = arr[..., bands]
        else:
            raise ValueError(
                "Cannot specify bands for a 2D array (no channel dimension)."
            )
    return arr


def spearman_rank_correlation_abs(arr1: np.ndarray, arr2: np.ndarray, bands=None) -> float:
    """
    Computes the Spearman rank correlation between two arrays,
    but uses the absolute values of each array before ranking.

    Parameters:
    -----------
    arr1, arr2 : np.ndarray
        Input arrays to compare. They must be the same shape (after band selection).
    bands : list or None
        Indices of the channels/bands to select. If None, all are used.

    Returns:
    --------
    float
        Spearman rank correlation coefficient in [-1, 1].
    """
    arr1_sel = _select_bands(arr1, bands)
    arr2_sel = _select_bands(arr2, bands)

    if arr1_sel.shape != arr2_sel.shape:
        raise ValueError("Input arrays must have the same shape after band selection.")

    # Flatten the arrays and take absolute values
    arr1_abs = np.abs(arr1_sel).flatten()
    arr2_abs = np.abs(arr2_sel).flatten()

    # Compute Spearman rank correlation
    correlation, _ = spearmanr(arr1_abs, arr2_abs)
    return correlation


def spearman_rank_correlation_noabs(arr1: np.ndarray, arr2: np.ndarray, bands=None) -> float:
    """
    Computes the Spearman rank correlation between two arrays (no absolute value).
    
    Parameters:
    -----------
    arr1, arr2 : np.ndarray
        Input arrays to compare. Must be the same shape (after band selection).
    bands : list or None
        Indices of the channels/bands to select. If None, all are used.

    Returns:
    --------
    float
        Spearman rank correlation coefficient in [-1, 1].
    """
    arr1_sel = _select_bands(arr1, bands)
    arr2_sel = _select_bands(arr2, bands)

    if arr1_sel.shape != arr2_sel.shape:
        raise ValueError("Input arrays must have the same shape after band selection.")

    arr1_flat = arr1_sel.flatten()
    arr2_flat = arr2_sel.flatten()

    correlation, _ = spearmanr(arr1_flat, arr2_flat)
    return correlation


def structural_similarity_index(
    arr1: np.ndarray,
    arr2: np.ndarray,
    bands=None,
    win_size=None
) -> float:
    """
    Computes the Structural Similarity Index (SSIM) between two images/arrays.
    Uses skimage.metrics.structural_similarity with channel_axis instead of 
    multichannel=True for 3D data (scikit-image >= 0.19).

    Parameters:
    -----------
    arr1, arr2 : np.ndarray
        Input arrays to compare. Must have the same shape after band selection.
    bands : list or None
        Indices of the channels/bands to select. If None, all are used.
    win_size : int or None
        Optional. The size of the Gaussian weighting window (must be odd). If None,
        skimage's default is used (typically 7 or 11). If you get a ValueError 
        about 'win_size exceeds image extent', pass a smaller odd value (e.g., 3 or 5).
    
    Returns:
    --------
    float
        SSIM value in [-1, 1]. (Usually, SSIM is in [0,1].)
    """
    arr1_sel = _select_bands(arr1, bands)
    arr2_sel = _select_bands(arr2, bands)

    if arr1_sel.shape != arr2_sel.shape:
        raise ValueError("Input arrays must have the same shape after band selection.")

    # If 3D, treat as multichannel by specifying channel_axis=-1
    channel_axis = -1 if arr1_sel.ndim == 3 else None

    ssim_value = structural_similarity(
        arr1_sel,
        arr2_sel,
        data_range=arr2_sel.max() - arr2_sel.min(),
        channel_axis=channel_axis,
        win_size=win_size
    )
    return ssim_value


def pearson_correlation_hog(arr1: np.ndarray, arr2: np.ndarray, bands=None) -> float:
    """
    Computes the Pearson Correlation between the HOG (Histogram of Oriented Gradients) 
    features of two images/arrays.

    Parameters:
    -----------
    arr1, arr2 : np.ndarray
        Input images to compare. Must be the same shape after band selection, 
        or at least valid for HOG extraction.
    bands : list or None
        Indices of the channels/bands to select. If None, all are used.

    Returns:
    --------
    float
        Pearson correlation coefficient (r) in [-1, 1].
    """
    arr1_sel = _select_bands(arr1, bands)
    arr2_sel = _select_bands(arr2, bands)

    if arr1_sel.shape != arr2_sel.shape:
        raise ValueError("Input arrays must have the same shape after band selection.")

    # Determine channel_axis for the HOG
    channel_axis = -1 if arr1_sel.ndim == 3 else None

    hog1 = hog(
        arr1_sel,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        channel_axis=channel_axis
    )

    hog2 = hog(
        arr2_sel,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        channel_axis=channel_axis
    )

    corr_coeff, _ = pearsonr(hog1, hog2)
    return corr_coeff


if __name__ == "__main__":
    # Simple usage examples / tests

    # Example 1: Single-band (2D) arrays
    img1 = np.random.random((64, 64))
    img2 = np.random.random((64, 64))

    print("Spearman (abs) single-band:", spearman_rank_correlation_abs(img1, img2))
    print("Spearman (no abs) single-band:", spearman_rank_correlation_noabs(img1, img2))
    print("SSIM single-band:", structural_similarity_index(img1, img2))
    print("HOG Pearson single-band:", pearson_correlation_hog(img1, img2))

    # Example 2: Multi-band (3D) arrays (e.g., 5-band image)
    img3 = np.random.random((64, 64, 5))
    img4 = np.random.random((64, 64, 5))
    selected_bands = [0, 1, 2]  # We'll only compare these three channels

    print("\nSpearman (abs) multi-band, first 3 channels:",
          spearman_rank_correlation_abs(img3, img4, bands=selected_bands))
    print("Spearman (no abs) multi-band, first 3 channels:",
          spearman_rank_correlation_noabs(img3, img4, bands=selected_bands))
    
    # If you see the 'win_size exceeds image extent' error, pass a smaller win_size:
    print("SSIM multi-band, first 3 channels:",
          structural_similarity_index(img3, img4, bands=selected_bands, win_size=7))

    print("HOG Pearson multi-band, first 3 channels:",
          pearson_correlation_hog(img3, img4, bands=selected_bands))


