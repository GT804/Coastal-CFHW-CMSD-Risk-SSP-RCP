# === 模块 0：导入依赖库 ===
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##=== 模块 1：读取 nc 文件与基础变量 ===
ds = xr.open_dataset("clipped_19912022_1.nc", engine="h5netcdf")
tmax = ds["tmax"]
tmin = ds["tmin"]
tp = ds["tp"]

baseline = ds.sel(time=slice("1991-01-01", "2011-12-31"))
study = ds.sel(time=slice("2012-01-01", "2022-12-31"))

tmax_study = study["tmax"]
tmin_study = study["tmin"]


# === 模块 2：定义热浪阈值计算函数 ===
def compute_thresholds(data):
    dayofyear = data.time.dt.dayofyear
    thresholds = []
    for doy in range(1, 367):
        daily = data.where(dayofyear == doy, drop=True)
        if daily.count().values.sum() == 0:
            dummy = xr.full_like(data.isel(time=0), np.nan)
            thresholds.append(dummy)
        else:
            q85 = daily.quantile(0.85, dim="time", skipna=True)
            thresholds.append(q85)
    result = xr.concat(thresholds, dim="dayofyear")
    result = result.assign_coords(dayofyear=np.arange(1, 367))
    return result

# === 模块 3：热浪日检测与频率计算 ===
tmax_thresh = compute_thresholds(baseline["tmax"])
tmin_thresh = compute_thresholds(baseline["tmin"])

doy_study = study.time.dt.dayofyear
doy_study = xr.where(doy_study == 366, 365, doy_study)

tmax_thresh_matched = xr.DataArray(tmax_thresh.sel(dayofyear=doy_study.values).values, dims=tmax_study.dims, coords=tmax_study.coords)
tmin_thresh_matched = xr.DataArray(tmin_thresh.sel(dayofyear=doy_study.values).values, dims=tmin_study.dims, coords=tmin_study.coords)

hw_day = (tmax_study > tmax_thresh_matched) | (tmin_study > tmin_thresh_matched)

def detect_heatwaves(hw_bool, min_len=3):
    hw_bool = hw_bool.fillna(False)
    shifted = [hw_bool.shift(time=-i, fill_value=False) for i in range(min_len)]
    result = shifted[0]
    for s in shifted[1:]:
        result = result & s
    return result

hw_event = detect_heatwaves(hw_day, min_len=3)
p_hw = hw_event.mean(dim="time")

# === 模块 4：洪水检测（WAP指数） ===
a = 0.9
N = 44
weights = (1 - a) * (a ** np.arange(N))

def compute_wap(tp):
    time_len = tp.sizes["time"]
    return xr.apply_ufunc(
        lambda x: np.convolve(x, weights, mode='valid'),
        tp,
        input_core_dims=[["time"]],
        output_core_dims=[["conv_time"]],
        output_sizes={"conv_time": time_len - N + 1},
        vectorize=True
    ).rename({"conv_time": "time"})



baseline_wap = compute_wap(baseline.tp).assign_coords(time=baseline.time[N-1:])
study_wap = compute_wap(study.tp).assign_coords(time=study.time[N-1:])

wap_thresh = baseline_wap.groupby("time.dayofyear").quantile(0.85, dim="time", skipna=True)
doy_study_wap = study.time[N-1:].dt.dayofyear
doy_study_wap = xr.where(doy_study_wap == 366, 365, doy_study_wap)

# 关键修复：不要使用 .values，保持 DataArray 匹配
wap_thresh_matched = wap_thresh.sel(dayofyear=doy_study_wap)
wap_thresh_matched = xr.DataArray(
    data=wap_thresh_matched.data,
    dims=study_wap.dims,
    coords=study_wap.coords
)

flood_event = (study_wap > wap_thresh_matched)
p_flood = flood_event.mean(dim="time")


# === 模块 5：复合事件识别与指标计算（逐像素滑窗判断） ===
from tqdm import tqdm

# 创建空的复合事件布尔数组
compound = xr.zeros_like(hw_event, dtype=bool)

# 滑动窗口长度（单位：天）
window_days = 7

# 获取纬度和经度范围（避免索引越界）
lat_size = hw_event.sizes['latitude']
lon_size = hw_event.sizes['longitude']

# 遍历每个格点
for lat_idx in tqdm(range(lat_size), desc="复合事件计算", position=0):
    for lon_idx in range(lon_size):
        try:
            hw_bool = hw_event[:, lat_idx, lon_idx].values
            flood_bool = flood_event[:, lat_idx, lon_idx].values

            for t in range(len(flood_bool)):
                if flood_bool[t]:
                    t_start = t
                    t_end = min(t + window_days, len(hw_bool))
                    if hw_bool[t_start:t_end].any():
                        compound[t, lat_idx, lon_idx] = True
        except IndexError:
            print(f"索引错误：lat_idx={lat_idx}, lon_idx={lon_idx}")
            continue

# 计算复合频率与指标
p_f_hw = compound.mean(dim="time")
rp_f_hw = 1 / (p_f_hw * 365)
lmf_f_hw = p_f_hw / (p_hw * p_flood)

# 复合事件布尔结果可选保存
# compound.astype("uint8").to_netcdf("compound_event_bool.nc")


# === 模块 6：导出年均频率与总体指标结果 ===
annual_hw = hw_event.groupby("time.year").sum(dim="time") / 365
annual_flood = flood_event.groupby("time.year").sum(dim="time") / 365
annual_comp = compound.groupby("time.year").sum(dim="time") / 365

mean_hw = annual_hw.mean(dim=["latitude", "longitude"])
mean_flood = annual_flood.mean(dim=["latitude", "longitude"])
mean_comp = annual_comp.mean(dim=["latitude", "longitude"])

summary = pd.DataFrame({
    "year": mean_hw.year.values,
    "P_HW": mean_hw.values,
    "P_Flood": mean_flood.values,
    "P_F-HW": mean_comp.values
})

summary.to_csv("P_F-HW_yearly.csv", index=False)

overall_result = pd.DataFrame({
    "P_HW": [float(p_hw.mean().values)],
    "P_Flood": [float(p_flood.mean().values)],
    "P_F-HW": [float(p_f_hw.mean().values)],
    "RP_F-HW": [float(rp_f_hw.mean().values)],
    "LMF_F_HW": [float(lmf_f_hw.mean().values)]
})

overall_result.to_csv("compound_event_summary.csv", index=False)

import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置全局字体为 Times New Roman
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['axes.unicode_minus'] = False  # 防止负号乱码

# 创建掩膜：有热或有洪水的地方
mask = ((p_hw > 0) | (p_flood > 0)).astype(int)

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Heatwave frequency plot
p_hw.plot(ax=axs[0], cmap="hot", cbar_kwargs={'label': 'Heatwave Frequency (P_HW)'})
axs[0].set_title("Heatwave Frequency (P_HW)", fontsize=13)

# Flood frequency plot
p_flood.plot(ax=axs[1], cmap="Blues", cbar_kwargs={'label': 'Flood Frequency (P_Flood)'})
axs[1].set_title("Flood Frequency (P_Flood)", fontsize=13)

# Composite event frequency plot: gray mask + overlay
mask.plot.imshow(
    ax=axs[2],
    cmap="Greys",
    vmin=0, vmax=1,
    add_colorbar=False
)
p_f_hw.plot.imshow(
    ax=axs[2],
    cmap="Purples",
    vmin=0, vmax=0.005,
    alpha=0.9,
    cbar_kwargs={'label': 'Compound Event Frequency (P_F-HW)'}
)
axs[2].set_title("Compound Event Frequency (P_F-HW)", fontsize=13)

# Axis labels
for ax in axs:
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置字体为 Times New Roman，避免中文乱码
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['axes.unicode_minus'] = False

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# 使用分位数设置 vmax（更自适应）
vmax_rp = float(rp_f_hw.quantile(0.98))
vmax_lmf = float(lmf_f_hw.quantile(0.98))

# 图1：RP_F_HW
mask.plot.imshow(
    ax=axs[0],
    cmap="Greys",
    vmin=0, vmax=1,
    add_colorbar=False,
    alpha=0.2
)
rp_f_hw.plot.imshow(
    ax=axs[0],
    cmap="YlOrRd",
    vmax=vmax_rp,
    alpha=0.95,
    cbar_kwargs={'label': 'Recurrence Period (RP-F-HW) [Years]'}
)
axs[0].set_title("Compound Event Recurrence Period (RP-F-HW)", fontsize=13)
axs[0].set_xlabel("Longitude", fontsize=12)
axs[0].set_ylabel("Latitude", fontsize=12)

# 图2：LMF_F_HW
mask.plot.imshow(
    ax=axs[1],
    cmap="Greys",
    vmin=0, vmax=1,
    add_colorbar=False,
    alpha=0.2
)
lmf_f_hw.plot.imshow(
    ax=axs[1],
    cmap="PuBuGn",
    vmax=vmax_lmf,
    vmin=0,
    alpha=0.95,
    cbar_kwargs={'label': 'Likelihood Multiplication Factor (LMF-F-HW)'}
)
axs[1].set_title("Likelihood Multiplication Factor (LMF-F-HW)", fontsize=13)
axs[1].set_xlabel("Longitude", fontsize=12)
axs[1].set_ylabel("Latitude", fontsize=12)

plt.tight_layout()
plt.show()



import rioxarray


# 设置空间参考（确保你的数据有正确经纬度坐标）
rp_f_hw.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
rp_f_hw.rio.write_crs("EPSG:4326", inplace=True)
rp_f_hw.rio.to_raster("RP_F_HW.tiff")

lmf_f_hw.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
lmf_f_hw.rio.write_crs("EPSG:4326", inplace=True)
lmf_f_hw.rio.to_raster("LMF_F_HW.tiff")


# === 模块 8：输出三个图为 GeoTIFF ===
for da, name in [(p_hw, "P_HW"), (p_flood, "P_Flood"), (p_f_hw, "P_F_HW")]:
    da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
    da.rio.write_crs("EPSG:4326", inplace=True)
    da.rio.to_raster(f"{name}.tiff")

    import os

    print("当前工作目录:", os.getcwd())  # 输出示例：/root/autodl-tmp/你的项目目录/
