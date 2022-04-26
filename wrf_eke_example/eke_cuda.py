import cupy as cp

from .eke import apply_filters, calc_uu_vv_2d, calc_eke_1d


def calc_averages(data, lat_index_north, lat_index_south, lon_index_west, lon_index_east,
                  time_slice=slice(0,-120), butterworth=True, butterworth_kwargs=None):
    """Compute the average Eddy Kinetic Energy (EKE) and total EKE for a given extent."""
    uu_vv_2d = []
    for d in data:
        if butterworth:
            u, v = apply_filters(
                d[0],
                d[1],
                time=time_slice,
                butterworth=butterworth,
                butterworth_kwargs=butterworth_kwargs)
        else:
            u = d[0]
            v = d[1]

        uv = calc_uu_vv_2d(
            u,
            v,
            lon_index_west,
            lon_index_east).persist()
        uu_vv_2d.append(uv)

    eke_2d = [0.5 * uv for uv in uu_vv_2d]
    eke_avg = cp.mean(eke_2d, axis=0)

    eke_1d = [calc_eke_1d(uv, lat_index_north, lat_index_south) for uv in uu_vv_2d]
    total_eke_avg = cp.mean(eke_1d, axis=0)

    return eke_avg, total_eke_avg
