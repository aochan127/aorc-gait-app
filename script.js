// Camera / Video / Image gait analysis (iPhone-safe)
// MediaPipe Tasks Vision: Pose Landmarker
import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

const videoEl = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const loadModelBtn = document.getElementById("loadModelBtn");
const startCamBtn = document.getElementById("startCamBtn");
const stopCamBtn = document.getElementById("stopCamBtn");
const viewSelect = document.getElementById("viewSelect");

const videoFile = document.getElementById("videoFile");
const playFileBtn = document.getElementById("playFileBtn");

const imageFile = document.getElementById("imageFile");
const analyzeImageBtn = document.getElementById("analyzeImageBtn");

const drawSkeleton = document.getElementById("drawSkeleton");
const showKeypoints = document.getElementById("showKeypoints");

const valgusL = document.getElementById("valgusL");
const valgusR = document.getElementById("valgusR");
const pelvicDrop = document.getElementById("pelvicDrop");
const stepWidth = document.getElementById("stepWidth");
const tibiaTilt = document.getElementById("tibiaTilt");
const statusBox = document.getElementById("status");

let poseLandmarkerVideo = null;   // runningMode: "VIDEO"
let poseLandmarkerImage = null;   // runningMode: "IMAGE"
let filesetResolver = null;

let stream = null;
let videoPlaying = false;
let lastTs = -1;

const metricsBuffer = { valgusL: [], valgusR: [], pelvicDrop: [], stepWidth: [], tibiaTilt: [] };
const MAX_BUF = 300;

function avg(a){ return a.length ? a.reduce((x,y)=>x+y,0)/a.length : NaN; }
function angleDeg(a,b,c){
  const v1={x:a.x-b.x,y:a.y-b.y}, v2={x:c.x-b.x,y:c.y-b.y};
  const dot=v1.x*v2.x+v1.y*v2.y, m1=Math.hypot(v1.x,v1.y), m2=Math.hypot(v2.x,v2.y);
  if(m1===0||m2===0) return NaN;
  let cos=dot/(m1*m2); cos=Math.max(-1,Math.min(1,cos));
  return Math.acos(cos)*180/Math.PI;
}
function tibiaTiltDeg(ankle,knee){ return Math.atan2(ankle.x-knee.x, Math.abs(ankle.y-knee.y))*180/Math.PI; }
function toPx(l,w,h){ return {x:l.x*w,y:l.y*h,z:l.z}; }
function setStatus(t){ statusBox.textContent = t; }

async function ensureFileset(){
  if(!filesetResolver){
    filesetResolver = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
  }
}
async function ensureVideoModel(){
  await ensureFileset();
  if(!poseLandmarkerVideo){
    poseLandmarkerVideo = await PoseLandmarker.createFromOptions(filesetResolver, {
      baseOptions:{ modelAssetPath:"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm/pose_landmarker_full.task", delegate:"GPU" },
      runningMode:"VIDEO",
      numPoses:1
    });
  }
}
async function ensureImageModel(){
  await ensureFileset();
  if(!poseLandmarkerImage){
    poseLandmarkerImage = await PoseLandmarker.createFromOptions(filesetResolver, {
      baseOptions:{ modelAssetPath:"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm/pose_landmarker_full.task", delegate:"GPU" },
      runningMode:"IMAGE",
      numPoses:1
    });
  }
}

document.addEventListener("DOMContentLoaded", () => {
  // 先に読み込んでおく（失敗しても後で再試行）
  ensureVideoModel().then(()=>setStatus("モデル読込完了")).catch(e=>setStatus("モデル読込エラー: "+e.message));
});

// ---------- Camera ----------
async function startCamera(){
  await ensureVideoModel();
  stream = await navigator.mediaDevices.getUserMedia({ video:{ facingMode:"environment" }, audio:false });
  videoEl.srcObject = stream;
  videoEl.playsInline = true; videoEl.setAttribute("playsinline","");
  await videoEl.play();
  videoPlaying = true;
  resizeCanvas();
  setStatus("カメラ動作中");
  startCamBtn.disabled = true; stopCamBtn.disabled = false;
  requestAnimationFrame(processFrame);
}
function stopCamera(){
  if(stream){ for(const t of stream.getTracks()) t.stop(); }
  videoEl.pause(); videoEl.srcObject=null; videoPlaying=false;
  startCamBtn.disabled=false; stopCamBtn.disabled=true;
  setStatus("停止");
}

// ---------- Video file ----------
videoFile?.addEventListener("change", async ()=>{ if(videoFile.files?.length) await playSelectedFile(); });
playFileBtn?.addEventListener("click", async ()=>{ if(!videoFile.files?.length) alert("動画を選択してください"); else await playSelectedFile(); });

async function playSelectedFile(){
  await ensureVideoModel();
  const file = videoFile.files[0];
  const url = URL.createObjectURL(file);

  videoEl.srcObject = null;
  videoEl.muted = true; videoEl.setAttribute("muted","muted");
  videoEl.playsInline = true; videoEl.setAttribute("playsinline",""); videoEl.setAttribute("webkit-playsinline","");
  videoEl.controls = true;
  videoEl.preload = "metadata";
  videoEl.src = url;

  setStatus("動画読込中…");

  videoEl.onloadedmetadata = async () => {
    resizeCanvas();
    setStatus("動画解析の準備完了（▶️で再生）");
    try {
      await videoEl.play();
      startVideoLoop();
    } catch(e) {
      setStatus("▶️ を押して再生してください");
      videoEl.addEventListener("play", startVideoLoop, { once:true });
    }
  };
}
function startVideoLoop(){ videoPlaying = true; setStatus("動画解析中…"); requestAnimationFrame(processFrame); }

// ---------- Image file ----------
analyzeImageBtn?.addEventListener("click", async ()=>{
  if(!imageFile.files?.length){ alert("画像を選択してください"); return; }
  await analyzeSelectedImage();
});
imageFile?.addEventListener("change", async ()=>{
  // 画像選択直後に自動解析したい場合はここで呼ぶ
  // await analyzeSelectedImage();
});

async function analyzeSelectedImage(){
  await ensureImageModel();
  const file = imageFile.files[0];

  // HEICはブラウザで読めないことがある
  if(file.type && !/^image\/(jpe?g|png|gif|webp)$/i.test(file.type)){
    setStatus("この画像形式は非対応の可能性があります（JPEG/PNGを推奨）");
  }

  const blobUrl = URL.createObjectURL(file);
  const img = new Image();
  img.onload = async () => {
    // 表示は動画領域を使わず、canvasに直接描画
    videoPlaying = false; // ループ停止
    resizeCanvasTo(img.naturalWidth, img.naturalHeight);
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    try{
      const res = await poseLandmarkerImage.detect(img);
      if(res && res.landmarks && res.landmarks.length){
        const lms = res.landmarks[0];
        drawResults(lms);
        // 単一画像のため“平均”は意味が薄いが、UIはそのまま更新
        computeMetrics(lms, canvas.width, canvas.height);
        updateMetricsUI();
        setStatus("画像解析完了");
      }else{
        setStatus("人物が検出できませんでした");
      }
    }catch(e){
      setStatus("画像解析エラー: " + e.message);
    } finally {
      URL.revokeObjectURL(blobUrl);
    }
  };
  img.onerror = ()=>{ setStatus("画像を読み込めませんでした"); URL.revokeObjectURL(blobUrl); };
  img.src = blobUrl;
}

// ---------- Drawing / Metrics ----------
function resizeCanvas(){
  const w = videoEl.videoWidth || canvas.width, h = videoEl.videoHeight || canvas.height;
  if(w && h){ canvas.width = w; canvas.height = h; }
}
function resizeCanvasTo(w,h){
  if(w && h){ canvas.width = w; canvas.height = h; }
}

function drawResults(lms){
  const du = new DrawingUtils(ctx);
  if(drawSkeleton.checked){ du.drawLandmarks(lms,{radius:3}); du.drawConnectors(lms, PoseLandmarker.POSE_CONNECTIONS); }
  else if(showKeypoints.checked){ du.drawLandmarks(lms,{radius:2}); }
}

function updateMetricsUI(){
  const vL=avg(metricsBuffer.valgusL), vR=avg(metricsBuffer.valgusR), pD=avg(metricsBuffer.pelvicDrop),
        sW=avg(metricsBuffer.stepWidth), tT=avg(metricsBuffer.tibiaTilt);
  valgusL.textContent=isNaN(vL)?"-":vL.toFixed(1);
  valgusR.textContent=isNaN(vR)?"-":vR.toFixed(1);
  pelvicDrop.textContent=isNaN(pD)?"-":pD.toFixed(1);
  stepWidth.textContent=isNaN(sW)?"-":sW.toFixed(2);
  tibiaTilt.textContent=isNaN(tT)?"-":tT.toFixed(1);
  document.querySelectorAll(".side-only").forEach(el=>{ el.style.display = (viewSelect.value==="side") ? "list-item":"none"; });
}

function computeMetrics(lms,w,h){
  const IDX={l_hip:23,r_hip:24,l_knee:25,r_knee:26,l_ankle:27,r_ankle:28};
  const L=(i)=>toPx(lms[i],w,h);
  const lh=L(IDX.l_hip), rh=L(IDX.r_hip), lk=L(IDX.l_knee), rk=L(IDX.r_knee), la=L(IDX.l_ankle), ra=L(IDX.r_ankle);
  const vl=angleDeg(lh,lk,la), vr=angleDeg(rh,rk,ra);
  if(!isNaN(vl)) metricsBuffer.valgusL.push(vl);
  if(!isNaN(vr)) metricsBuffer.valgusR.push(vr);
  const hipW=Math.max(1,Math.hypot(lh.x-rh.x,lh.y-rh.y));
  const dy=(lh.y-rh.y);
  const angle=Math.atan2(Math.abs(dy),hipW)*180/Math.PI;
  metricsBuffer.pelvicDrop.push(angle);
  metricsBuffer.stepWidth.push(Math.abs(la.x-ra.x)/hipW);
  const tib=Math.abs((la.y>ra.y)?tibiaTiltDeg(la,lk):tibiaTiltDeg(ra,rk));
  if(!isNaN(tib)) metricsBuffer.tibiaTilt.push(tib);
  for(const k in metricsBuffer){ if(metricsBuffer[k].length>MAX_BUF) metricsBuffer[k].shift(); }
}

// ---------- Loop ----------
async function processFrame(ts){
  if(!videoPlaying) return;
  if(lastTs===ts){ requestAnimationFrame(processFrame); return; }
  lastTs=ts;
  resizeCanvas();
  ctx.clearRect(0,0,canvas.width,canvas.height);
  try{
    if(poseLandmarkerVideo){
      const res = await poseLandmarkerVideo.detectForVideo(videoEl, ts);
      if(res && res.landmarks && res.landmarks.length){
        const lms = res.landmarks[0];
        drawResults(lms);
        computeMetrics(lms, canvas.width, canvas.height);
        updateMetricsUI();
      }
    }
  }catch(e){ setStatus("推論エラー: "+e.message); }
  requestAnimationFrame(processFrame);
}

// ---------- Events ----------
loadModelBtn.addEventListener("click", async ()=>{
  try{ await Promise.all([ensureVideoModel(), ensureImageModel()]); setStatus("モデル読込完了"); }
  catch(e){ setStatus("モデル読込エラー: "+e.message); }
});
startCamBtn.addEventListener("click", startCamera);
stopCamBtn.addEventListener("click", stopCamera);
playFileBtn?.addEventListener("click", async ()=>{ if(!videoFile.files?.length) alert("動画を選択してください"); });
