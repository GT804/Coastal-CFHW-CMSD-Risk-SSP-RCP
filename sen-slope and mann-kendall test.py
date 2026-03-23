import xarray as xr
import numpy as np
from pymannkendall import original_test
import rioxarray

# === 模块1：数据预处理（年值计算） ===
ds = xr.open_dataset(r"C:\Users\dell\*****\CEFHW\clipped_19912022_1.nc", engine="h5netcdf")

# 检查原始数据维度名称（关键修改点）
print("原始数据集维度结构：\n", ds.dims)
print("原始数据坐标名称：\n", ds.coords)

# 计算年平均值（温度）和年总量（降水）
tmax_annual = ds["tmax"].resample(time='Y').mean(dim='time')
tmin_annual = ds["tmin"].resample(time='Y').mean(dim='time')
tp_annual = ds["tp"].resample(time='Y').sum(dim='time') * 1000  # 转换为毫米（假设原始单位为米）

# 构建包含所有年份的数据集（1991-2022共32年）
years = np.arange(1991, 2023)
tmax_annual = tmax_annual.assign_coords(time=years)
tmin_annual = tmin_annual.assign_coords(time=years)
tp_annual = tp_annual.assign_coords(time=years)

# === 模块2：定义趋势分析函数 ===
def theil_sen(y):
    """计算Sen's slope（添加NaN处理）"""
    if np.isnan(y).all():
        return np.nan
    y_clean = y[~np.isnan(y)]
    n = len(y_clean)
    if n < 2:
        return 0.0
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            slope = (y_clean[j] - y_clean[i]) / (j - i)
            if not np.isnan(slope):
                slopes.append(slope)
    return np.median(slopes) if slopes else 0.0

def mk_trend(y):
    """执行MK检验并返回趋势方向与显著性（增强异常处理）"""
    if np.isnan(y).all():
        return np.array([np.nan, np.nan])
    try:
        result = original_test(y[~np.isnan(y)])
        trend = 1 if result.trend == 'increasing' else -1 if result.trend == 'decreasing' else 0
        sig = 1 if result.p < 0.05 else 0
        return np.array([trend, sig])
    except Exception as e:
        print(f"MK检验出错：{str(e)}")
        return np.array([np.nan, np.nan])

# === 模块3：逐变量计算趋势与显著性 ===
variables = {
    "tmax": tmax_annual,
    "tmin": tmin_annual,
    "tp": tp_annual
}

for var_name, da_annual in variables.items():
    print(f"\n正在处理 {var_name}...")

    # 计算Sen's slope
    sen_slope = xr.apply_ufunc(
        theil_sen,
        da_annual,
        input_core_dims=[["time"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float]
    )

    # 计算MK趋势与显著性
    mk_result = xr.apply_ufunc(
        mk_trend,
        da_annual,
        input_core_dims=[["time"]],
        output_core_dims=[["stat"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float]
    )

    # 拆分结果并处理数据类型
    trend = mk_result.sel(stat=0).astype(np.int8)  # 使用int8节省空间
    significance = mk_result.sel(stat=1).astype(np.uint8)  # 使用uint8避免符号问题

    # === 模块4：输出GeoTIFF（关键修改点） ===
    # 确保所有数据集具有正确的空间维度名称（根据实际情况修改）
    x_dim = "lon" if "lon" in da_annual.dims else "longitude"
    y_dim = "lat" if "lat" in da_annual.dims else "latitude"

    # 趋势斜率（强制转换为float32）
    sen_slope = sen_slope.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)
    sen_slope.rio.write_crs("EPSG:4326", inplace=True)
    sen_slope.astype(np.float32).rio.to_raster(f"{var_name}_sen_slope.tif")

    # 显著性标记（强制转换为uint8）
    significance = significance.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)
    significance.rio.write_crs("EPSG:4326", inplace=True)
    significance.astype(np.uint8).rio.to_raster(f"{var_name}_mk_significance.tif")

    # 带显著性标记的斜率（处理NaN）
    sen_slope_sig = sen_slope.where(significance == 1)
    sen_slope_sig.astype(np.float32).rio.to_raster(f"{var_name}_sen_slope_sig_only.tif")

print("\n==== 处理完成 ====")
