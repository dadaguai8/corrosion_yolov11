const state = {
  source: "image",
  imageFile: null,
  imageResult: null,
  folderFiles: [],
  folderResults: [],
  folderIndex: 0,
  videoFile: null,
  stream: null,
  detecting: false,
  realtimeLoopId: null,
  videoResult: null,
  videoObjectUrl: null,
  folderBatchSummary: null,
  lastResult: null,
};

const $ = (id) => document.getElementById(id);
const canvas = $("resultCanvas");
const ctx = canvas.getContext("2d");
const video = $("videoPreview");
const REALTIME_FRAME_DELAY_MS = 150;

const sourceLabels = {
  image: "图片检测",
  folder: "文件夹检测",
  video: "视频检测",
  camera: "摄像头检测",
};

function setStatus(text, type = "") {
  const badge = $("statusBadge");
  badge.textContent = text;
  badge.className = `status-badge ${type}`.trim();
}

function setBusy(isBusy, text = "开始检测") {
  state.detecting = isBusy;
  $("detectButton").textContent = isBusy ? "停止检测" : text;
  document.querySelectorAll("button, input, select").forEach((el) => {
    if (!["detectButton", "saveButton", "reportButton"].includes(el.id)) {
      el.disabled = isBusy && ["video", "camera"].includes(state.source);
    }
  });
  $("saveButton").disabled = !state.lastResult?.downloadUrl;
  $("reportButton").disabled = !state.lastResult?.reportUrl;
}

function params() {
  return {
    conf: $("confidence").value,
    iou: $("iou").value,
  };
}

function updateRangeText() {
  $("confText").textContent = Number($("confidence").value).toFixed(2);
  $("iouText").textContent = Number($("iou").value).toFixed(2);
}

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function showEmpty(show) {
  $("emptyState").style.display = show ? "grid" : "none";
}

function isImageFile(file) {
  return Boolean(file?.type?.startsWith("image/") || /\.(jpe?g|png|bmp|gif|webp)$/i.test(file?.name || ""));
}

function showCanvasPreview() {
  video.pause();
  video.srcObject = null;
  video.style.display = "none";
  canvas.style.display = "block";
}

function syncCanvasToVideoFrame() {
  if (!video.videoWidth || !video.videoHeight) return;
  if (canvas.width === video.videoWidth && canvas.height === video.videoHeight) return;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

function drawImageUrl(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      clearCanvas();
      const scale = Math.min(canvas.width / image.width, canvas.height / image.height);
      const width = image.width * scale;
      const height = image.height * scale;
      const x = (canvas.width - width) / 2;
      const y = (canvas.height - height) / 2;
      ctx.drawImage(image, x, y, width, height);
      showEmpty(false);
      resolve();
    };
    image.onerror = reject;
    image.src = url;
  });
}

function drawFilePreview(file) {
  if (!file) return;
  showCanvasPreview();
  const url = URL.createObjectURL(file);
  drawImageUrl(url).finally(() => URL.revokeObjectURL(url));
}

function updateSummary(result) {
  const safe = result || {
    hasDetectionResult: false,
    count: 0,
    averageConfidence: 0,
    typeCount: 0,
    elapsedMs: 0,
    areaRatio: 0,
    severity: "未检测",
    counts: { crevice: 0, pitting: 0, uniform: 0 },
    rows: [],
  };
  const hasDetectionResult = Boolean(result);

  $("countValue").textContent = hasDetectionResult && Number(safe.count) === 0 ? "无腐蚀" : (safe.count || 0);
  $("avgValue").textContent = `${safe.averageConfidence || 0}%`;
  $("typeValue").textContent = `${safe.typeCount || 0}类`;
  $("timeValue").textContent = `${safe.elapsedMs || 0} ms`;
  $("areaValue").textContent = `${safe.areaRatio || 0}%`;
  $("severityValue").textContent = safe.severity || "未检测";

  const counts = safe.counts || {};
  const max = Math.max(1, counts.crevice || 0, counts.pitting || 0, counts.uniform || 0);
  [["crevice", "crevice"], ["pitting", "pitting"], ["uniform", "uniform"]].forEach(([key, id]) => {
    const value = counts[key] || 0;
    $(`${id}Count`).textContent = value;
    $(`${id}Bar`).style.width = `${(value / max) * 100}%`;
  });

  const rows = safe.rows || [];
  $("resultRows").innerHTML = rows.length
    ? rows.map((row) => `<tr><td>${escapeHtml(row.className)}</td><td>${row.confidence}%</td><td>${row.areaRatio}%</td><td>${escapeHtml(row.position)}</td></tr>`).join("")
    : `<tr><td colspan="4">暂无检测结果</td></tr>`;

  $("saveButton").disabled = !safe.downloadUrl;
  $("reportButton").disabled = !safe.reportUrl;
  state.lastResult = safe.downloadUrl || safe.reportUrl ? safe : state.lastResult;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }[char]));
}

function formWithParams() {
  const form = new FormData();
  const p = params();
  form.append("conf", p.conf);
  form.append("iou", p.iou);
  return form;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || `请求失败：${response.status}`);
  }
  return payload;
}

async function loadModelInfo() {
  try {
    const data = await fetchJson("/api/model");
    const modelSelect = $("modelSelect");
    modelSelect.innerHTML = "";
    const models = data.models?.length
      ? data.models
      : [{ path: data.modelPath, name: data.modelName || filenameFromPath(data.modelPath), active: true }];
    models.forEach((item) => {
      const option = document.createElement("option");
      option.value = item.path;
      option.textContent = item.name || filenameFromPath(item.path);
      option.selected = Boolean(item.active);
      option.title = item.path;
      modelSelect.appendChild(option);
    });
    if (!data.loaded) {
      const option = document.createElement("option");
      option.textContent = `模型加载失败：${data.error}`;
      option.disabled = true;
      modelSelect.appendChild(option);
    }
    $("mapValue").textContent = data.map50 || "--";
    $("fpsValue").textContent = data.fps || "--";
    setStatus(data.loaded ? "后端已连接" : "模型异常", data.loaded ? "ready" : "error");
  } catch (error) {
    setStatus("后端未连接", "error");
    $("modelSelect").innerHTML = `<option>${escapeHtml(error.message)}</option>`;
  }
}

function filenameFromPath(path) {
  return String(path || "").split(/[\\/]/).pop() || "未知模型";
}

async function uploadModel(file) {
  if (!file) return;
  setStatus("正在加载模型", "busy");
  const form = new FormData();
  form.append("model", file);
  try {
    await fetchJson("/api/model", { method: "POST", body: form });
    await loadModelInfo();
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function switchModel(modelPath) {
  if (!modelPath) return;
  setStatus("正在切换模型", "busy");
  const form = new FormData();
  form.append("modelPath", modelPath);
  try {
    await fetchJson("/api/model", { method: "POST", body: form });
    await loadModelInfo();
    setStatus("模型切换成功", "ready");
  } catch (error) {
    setStatus(error.message, "error");
    await loadModelInfo();
  }
}

async function detectImage() {
  if (!state.imageFile) {
    setStatus("请先选择图片", "error");
    return;
  }
  setStatus("正在检测图片", "busy");
  const form = formWithParams();
  form.append("image", state.imageFile);
  const payload = await fetchJson("/api/detect/image", { method: "POST", body: form });
  state.imageResult = payload.result;
  updateSummary(payload.result);
  showCanvasPreview();
  await drawImageUrl(payload.result.resultUrl);
  updateSingleImagePager(true);
  setStatus("检测完成", "ready");
}

function hidePreviewNavigator() {
  $("folderNavigator").hidden = true;
  $("folderPosition").textContent = "0 / 0";
}

function updateSingleImagePager(hasImage) {
  const shouldShow = hasImage && state.source === "image";
  if (!shouldShow) {
    hidePreviewNavigator();
    return;
  }
  $("folderNavigator").hidden = false;
  $("folderPosition").textContent = "1 / 1";
}

async function detectFolder() {
  if (!state.folderFiles.length) {
    setStatus("请先选择图片文件夹", "error");
    return;
  }
  const total = state.folderFiles.length;
  let failed = 0;
  state.folderResults = [];
  state.folderBatchSummary = null;
  state.folderIndex = 0;
  setStatus("已检测 0", "busy");

  for (let index = 0; index < state.folderFiles.length; index += 1) {
    const file = state.folderFiles[index];
    const form = formWithParams();
    form.append("image", file, file.webkitRelativePath || file.name);
    try {
      const payload = await fetchJson("/api/detect/image", { method: "POST", body: form });
      state.folderResults.push(payload.result);
      state.folderIndex = state.folderResults.length - 1;
      updateSummary(buildFolderAggregate(state.folderResults));
      updateFolderView();
    } catch (error) {
      failed += 1;
      console.error(error);
    }
    setStatus(`已检测 ${state.folderResults.length}`, "busy");
    await new Promise((resolve) => window.setTimeout(resolve, 0));
  }

  if (!state.folderResults.length) {
    throw new Error("文件夹检测失败，未生成有效结果");
  }

  const finalize = await fetchJson("/api/detect/batch/finalize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids: state.folderResults.map((item) => item.id) }),
  });
  state.folderBatchSummary = finalize.result;
  state.folderIndex = 0;
  updateFolderView();
  setBatchActions(finalize.result);
  setStatus(failed ? `已检测 ${state.folderResults.length}，失败 ${failed}` : `已检测 ${state.folderResults.length}`, "ready");
}

function buildFolderAggregate(items) {
  const counts = { crevice: 0, pitting: 0, uniform: 0 };
  const rows = [];
  let count = 0;
  let area = 0;
  let elapsed = 0;
  const confidences = [];

  items.forEach((item) => {
    count += item.count || 0;
    area += item.areaRatio || 0;
    elapsed += item.elapsedMs || 0;
    rows.push(...(item.rows || []));
    (item.rows || []).forEach((row) => confidences.push(row.confidence || 0));
    Object.keys(counts).forEach((key) => {
      counts[key] += Number(item.counts?.[key] || 0);
    });
  });

  const areaRatio = items.length ? Number((area / items.length).toFixed(2)) : 0;
  return {
    count,
    averageConfidence: confidences.length ? Number((confidences.reduce((sum, value) => sum + value, 0) / confidences.length).toFixed(2)) : 0,
    typeCount: Object.values(counts).filter((value) => value > 0).length,
    elapsedMs: elapsed,
    areaRatio,
    severity: count ? (areaRatio >= 10 ? "严重" : areaRatio >= 3 ? "中等" : "轻微") : "未检测",
    counts,
    rows,
  };
}

function setBatchActions(summary) {
  if (!summary) return;
  state.lastResult = summary;
  $("saveButton").disabled = !summary.downloadUrl;
  $("reportButton").disabled = !summary.reportUrl;
}

function updateFolderSelectionView() {
  if (state.source !== "folder") {
    hidePreviewNavigator();
    return;
  }
  const total = state.folderFiles.length;
  $("folderNavigator").hidden = total <= 0;
  $("folderPosition").textContent = `${total ? state.folderIndex + 1 : 0} / ${total}`;

  if (!total) {
    $("fileName").textContent = "等待输入";
    showEmpty(true);
    return;
  }

  const file = state.folderFiles[state.folderIndex];
  $("fileName").textContent = file?.webkitRelativePath || file?.name || `第 ${state.folderIndex + 1} 张`;
  drawFilePreview(file);
}

function updateFolderView() {
  if (state.source !== "folder") {
    hidePreviewNavigator();
    return;
  }
  const total = state.folderResults.length;
  $("folderNavigator").hidden = total <= 0;
  $("folderPosition").textContent = `${total ? state.folderIndex + 1 : 0} / ${total}`;
  if (!total) return;
  const item = state.folderResults[state.folderIndex];
  $("fileName").textContent = item.sourceName || `第 ${state.folderIndex + 1} 张`;
  updateSummary(item);
  setBatchActions(state.folderBatchSummary);
  showCanvasPreview();
  drawImageUrl(item.resultUrl);
}

function clearVideoResultState(resetPanel = false) {
  state.videoResult = null;
  if (!resetPanel) return;
  state.lastResult = null;
  updateSummary(null);
  $("saveButton").disabled = true;
  $("reportButton").disabled = true;
}

function clearVideoFilePlayback() {
  video.pause();
  if (state.videoObjectUrl) {
    URL.revokeObjectURL(state.videoObjectUrl);
    state.videoObjectUrl = null;
  }
  video.removeAttribute("src");
  video.removeAttribute("autoplay");
  video.srcObject = null;
  video.load();
}

function resetVideoSelection() {
  state.videoFile = null;
  const input = $("videoInput");
  if (input) input.value = "";
}

function setVideoPreviewFile(file) {
  if (state.stream) stopCamera();
  if (state.videoObjectUrl) URL.revokeObjectURL(state.videoObjectUrl);
  state.videoObjectUrl = URL.createObjectURL(file);
  video.pause();
  video.srcObject = null;
  video.autoplay = false;
  video.removeAttribute("autoplay");
  video.removeAttribute("src");
  video.load();
  video.src = state.videoObjectUrl;
  video.controls = true;
  video.style.display = "block";
  canvas.style.display = "none";
  showEmpty(false);
  video.addEventListener("loadedmetadata", () => {
    try {
      video.currentTime = 0;
    } catch (error) {
      console.debug("Unable to reset video time before preview", error);
    }
  }, { once: true });
  video.load();
}

function metricForVideoTime(currentTime) {
  const metrics = state.videoResult?.frameMetrics || [];
  if (!metrics.length) return null;
  let selected = metrics[0];
  for (const metric of metrics) {
    if (Number(metric.timeSec || 0) <= currentTime + 0.001) selected = metric;
    else break;
  }
  return selected;
}

function showVideoFrameMetrics(currentTime) {
  const metric = metricForVideoTime(currentTime);
  if (!metric) return;
  updateSummary(metric);
  state.lastResult = state.videoResult;
  $("saveButton").disabled = !state.videoResult?.downloadUrl;
  $("reportButton").disabled = !state.videoResult?.reportUrl;
}

function syncVideoMetrics() {
  if (!state.videoResult || state.source !== "video") return;
  showVideoFrameMetrics(video.currentTime);
}

function playResultVideoOnce() {
  const playback = video.play();
  if (!playback || typeof playback.catch !== "function") return;
  playback.catch(() => {
    setStatus("视频检测完成，点击视频播放", "ready");
  });
}

function prepareResultVideoPlayback(resultVideoUrl) {
  video.pause();
  video.srcObject = null;
  video.muted = true;
  video.defaultMuted = true;
  video.volume = 0;
  video.autoplay = true;
  video.playsInline = true;
  video.preload = "auto";
  video.setAttribute("muted", "");
  video.setAttribute("autoplay", "");
  video.setAttribute("playsinline", "");
  video.setAttribute("webkit-playsinline", "");
  video.src = resultVideoUrl;
  video.controls = true;
  video.style.display = "block";
  ["loadeddata", "canplay", "canplaythrough"].forEach((eventName) => {
    video.addEventListener(eventName, playResultVideoOnce, { once: true });
  });
  video.load();
  window.requestAnimationFrame(playResultVideoOnce);
  window.setTimeout(playResultVideoOnce, 250);
  playResultVideoOnce();
}

async function detectVideoFile() {
  hidePreviewNavigator();
  if (state.detecting) {
    setStatus("视频正在处理中", "busy");
    return;
  }

  if (!state.videoFile) {
    setStatus("请先选择视频", "error");
    return;
  }

  video.style.display = "block";
  canvas.style.display = "none";
  clearCanvas();
  showEmpty(false);
  setBusy(true);
  setStatus("正在处理完整视频", "busy");
  const form = formWithParams();
  form.append("video", state.videoFile);
  try {
    const payload = await fetchJson("/api/detect/video", { method: "POST", body: form });
    state.videoResult = payload.result;
    showVideoFrameMetrics(0);
    prepareResultVideoPlayback(payload.result.resultVideoUrl);
    $("fileName").textContent = payload.result.sourceName || state.videoFile.name;
    setStatus("视频检测完成", "ready");
  } finally {
    setBusy(false);
  }
}

async function startRealtimeDetection() {
  hidePreviewNavigator();
  if (state.detecting) {
    stopRealtime();
    return;
  }

  if (!state.stream) {
    await startCamera();
    if (!state.stream) return;
  }

  video.style.display = "block";
  canvas.style.display = "block";
  showEmpty(false);
  setBusy(true);
  setStatus("实时检测中", "busy");
  scheduleNextFrame(0);
}

function stopRealtime() {
  if (state.realtimeLoopId) window.clearTimeout(state.realtimeLoopId);
  state.realtimeLoopId = null;
  setBusy(false);
  setStatus("已停止检测", "ready");
}

function scheduleNextFrame(delay = REALTIME_FRAME_DELAY_MS) {
  if (!state.detecting) return;
  if (state.realtimeLoopId) window.clearTimeout(state.realtimeLoopId);
  state.realtimeLoopId = window.setTimeout(captureAndDetectFrame, delay);
}

let frameBusy = false;
async function captureAndDetectFrame() {
  state.realtimeLoopId = null;
  if (!state.detecting) return;
  if (frameBusy) {
    scheduleNextFrame(REALTIME_FRAME_DELAY_MS);
    return;
  }
  if (video.readyState < 2) {
    scheduleNextFrame(REALTIME_FRAME_DELAY_MS);
    return;
  }
  frameBusy = true;
  try {
    syncCanvasToVideoFrame();
    const frameCanvas = document.createElement("canvas");
    frameCanvas.width = video.videoWidth || 960;
    frameCanvas.height = video.videoHeight || 540;
    frameCanvas.getContext("2d").drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);
    const image = frameCanvas.toDataURL("image/jpeg", 0.82);
    const payload = await fetchJson("/api/detect/frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image, name: `${state.source}-frame.jpg`, ...params() }),
    });
    updateSummary(payload.result);
    await drawImageUrl(payload.result.resultImage);
  } catch (error) {
    setStatus(error.message, "error");
    stopRealtime();
  } finally {
    frameBusy = false;
    if (state.detecting) {
      scheduleNextFrame(REALTIME_FRAME_DELAY_MS);
    }
  }
}

async function startCamera() {
  try {
    hidePreviewNavigator();
    clearVideoFilePlayback();
    state.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = state.stream;
    video.autoplay = true;
    video.controls = false;
    await video.play();
    $("fileName").textContent = "浏览器摄像头";
    setStatus("摄像头已启动", "ready");
  } catch (error) {
    setStatus(`摄像头启动失败：${error.message}`, "error");
  }
}

function stopCamera() {
  video.pause();
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
  video.srcObject = null;
  video.removeAttribute("autoplay");
  video.load();
  video.style.display = "none";
  stopRealtime();
}

async function handleDetect() {
  try {
    if (state.source === "image") await detectImage();
    if (state.source === "folder") await detectFolder();
    if (state.source === "video") await detectVideoFile();
    if (state.source === "camera") await startRealtimeDetection();
  } catch (error) {
    setStatus(error.message, "error");
    setBusy(false);
  }
}

function switchSource(source) {
  if (state.detecting) stopRealtime();
  if (state.source === "camera" && source !== "camera") stopCamera();
  if (source !== "video") {
    clearVideoResultState();
    clearVideoFilePlayback();
    resetVideoSelection();
  }
  state.source = source;
  $("sourceLabel").textContent = sourceLabels[source];
  $("fileName").textContent = "等待输入";
  hidePreviewNavigator();
  video.style.display = source === "video" || source === "camera" ? "block" : "none";
  canvas.style.display = source === "video" ? "none" : "block";
  clearCanvas();
  showEmpty(source !== "video" && source !== "camera");
  if (source === "video" || source === "camera") {
    hidePreviewNavigator();
  }
  if (source === "image") {
    updateSingleImagePager(Boolean(state.imageFile));
    if (state.imageResult) {
      $("fileName").textContent = state.imageResult.sourceName || state.imageFile?.name || "等待输入";
      updateSummary(state.imageResult);
      showCanvasPreview();
      drawImageUrl(state.imageResult.resultUrl);
    } else if (state.imageFile) {
      $("fileName").textContent = state.imageFile.name;
      drawFilePreview(state.imageFile);
    }
  }
  if (source === "folder") {
    if (state.folderResults.length) updateFolderView();
    else updateFolderSelectionView();
  }
  document.querySelectorAll(".tab-button").forEach((button) => button.classList.toggle("active", button.dataset.source === source));
  document.querySelectorAll(".source-pane").forEach((pane) => pane.classList.toggle("active", pane.dataset.pane === source));
}

function initEvents() {
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", () => switchSource(button.dataset.source));
  });

  $("confidence").addEventListener("input", updateRangeText);
  $("iou").addEventListener("input", updateRangeText);
  $("detectButton").addEventListener("click", handleDetect);
  $("modelSelect").addEventListener("change", (event) => switchModel(event.target.value));
  $("modelFile").addEventListener("change", (event) => uploadModel(event.target.files[0]));

  $("imageInput").addEventListener("change", (event) => {
    state.imageFile = event.target.files[0] || null;
    state.imageResult = null;
    if (state.imageFile) {
      $("fileName").textContent = state.imageFile.name;
      drawFilePreview(state.imageFile);
      updateSingleImagePager(true);
      setStatus("图片已选择", "ready");
    } else {
      updateSingleImagePager(false);
    }
  });

  $("folderInput").addEventListener("change", (event) => {
    state.folderFiles = Array.from(event.target.files || []).filter(isImageFile);
    state.folderResults = [];
    state.folderBatchSummary = null;
    state.folderIndex = 0;
    updateFolderSelectionView();
    setStatus(state.folderFiles.length ? "文件夹已选择" : "未选择图片", state.folderFiles.length ? "ready" : "error");
  });

  $("videoInput").addEventListener("change", async (event) => {
    hidePreviewNavigator();
    clearVideoResultState(true);
    state.videoFile = event.target.files[0] || null;
    if (!state.videoFile) return;
    setVideoPreviewFile(state.videoFile);
    video.play().catch(() => {});
    $("fileName").textContent = state.videoFile.name;
    setStatus("视频已载入", "ready");
  });

  $("cameraButton").addEventListener("click", async () => {
    if (state.stream) stopCamera();
    else await startCamera();
  });

  video.addEventListener("loadedmetadata", () => {
    if (!state.videoResult || state.source !== "video") return;
    showVideoFrameMetrics(0);
    playResultVideoOnce();
  });
  video.addEventListener("timeupdate", syncVideoMetrics);

  $("prevImageButton").addEventListener("click", () => {
    if (state.source === "image") {
      updateSingleImagePager(Boolean(state.imageFile));
      return;
    }
    const total = state.folderResults.length || state.folderFiles.length;
    if (!total) return;
    state.folderIndex = (state.folderIndex - 1 + total) % total;
    if (state.folderResults.length) updateFolderView();
    else updateFolderSelectionView();
  });

  $("nextImageButton").addEventListener("click", () => {
    if (state.source === "image") {
      updateSingleImagePager(Boolean(state.imageFile));
      return;
    }
    const total = state.folderResults.length || state.folderFiles.length;
    if (!total) return;
    state.folderIndex = (state.folderIndex + 1) % total;
    if (state.folderResults.length) updateFolderView();
    else updateFolderSelectionView();
  });

  $("saveButton").addEventListener("click", () => {
    if (state.lastResult?.downloadUrl) window.open(state.lastResult.downloadUrl, "_blank");
  });

  $("reportButton").addEventListener("click", () => {
    if (state.lastResult?.reportUrl) window.open(state.lastResult.reportUrl, "_blank");
  });
}

initEvents();
updateRangeText();
updateSummary(null);
loadModelInfo();
