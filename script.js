// Camera & Video gait analysis (iPhone-safe)
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

const drawSkeleton = document.getElementById("drawSkeleton");
const showKeypoints = document.getElementById("showKeypoints");

const valgusL = document.getElementById("valgusL");
const valgusR = document.getElementById("valgusR");
const pelvicDrop = document.getElementById("pelvicDrop");
const stepWidth = document.getElementById("stepWidth");
const tibiaTilt = document.getElementById("tibiaTilt");
const statusBox = document.getElementById("status");

let poseLandmarker = null;
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
function tibiaTiltDeg(ankle,knee){ const dx=ankle.x-knee.x, dy=ankle.y-knee.y; return Math.atan2(dx,Math.abs(dy))*180/Math.PI; }
function toPx(l,w,h){ return {x:l.x*w,y:l.y*h,z:l.z}; }

async function loadModel(){
  setStatus("モデル読込中…");
  const fr = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
  poseLandmarker = await PoseLandmarker.createFromOptions(fr, {
    baseOptions:{ modelAssetPath:"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm/pose_landmarker_full.task", delegate:"GPU" },
    runningMode:"VIDEO",
    numPoses:1
  });
  setStatus("モデル読込完了");
}
document.addEventListener("DOMContentLoaded", () => loadModel().catch(e=>setStatus("モデル読込エラー: "+e.message)));

function setStatus(t){ statusBox.textContent = t; }

// ------- Camera -------
async function startCamera(){
  if(!poseLandmarker) await loadModel();
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

// ------- Video file (iPhone対策: file選択直後に処理開始) -------
videoFile?.addEventListener("change", async () => {
  if(!videoFile.files?.length) return;
  await playSelectedFile();
});
playFileBtn?.addEventListener("click", async () => {
  if(!videoFile.files?.length){ alert("動画を選択してください"); return; }
  await playSelectedFile();
});

async function playSelectedFile(){
  if(!poseLandmarker) await loadModel();
  const file = videoFile.files[0];
  const url = URL.createObjectURL(file);

  // iOS向け属性（HTML側にも付けてるが保険で再設定）
  videoEl.srcObject = null;
  videoEl.muted = true; videoEl.setAttribute("muted","muted");
  videoEl.playsInline = true; videoEl.setAttribute("playsinline",""); videoEl.setAttribute("webkit-playsinline","");
  videoEl.controls = true;
  videoEl.preload = "metadata";
  videoEl.src = url;

  setStatus("動画読込中…");

  // デバッグ: 再生/エラーイベントをUIに出す
  const log = (ev)=> setStatus(`動画イベント: ${ev.type}`);
  ["loadedmetadata","loadeddata","canplay","play","pause","error","stalled","waiting"].forEach(ev=>videoEl.addEventListener(ev,log,{once:false}));

  videoEl.onloadedmetadata = async () => {
    resizeCanvas();
    setStatus("動画解析の準備完了（▶️で再生）");
    try {
      // ファイル選択という“ユーザー操作直後”なので再生できる可能性が高い
      await videoEl.play();
      startVideoLoop();
    } catch(e) {
      // 自動再生が弾かれたら手動▶️で開始
      setStatus("▶️ を押して再生してください");
      videoEl.addEventListener("play", startVideoLoop, { once:true });
    }
  };
}

function startVideoLoop(){ videoPlaying = true; setStatus("動画解析中…"); requestAnimationFrame(processFrame); }

function resizeCanvas(){
  const w = videoEl.videoWidth || canvas.width, h = videoEl.videoHeight || canvas.height;
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

async function processFrame(ts){
  if(!videoPlaying) return;
  if(lastTs===ts){ requestAnimationFrame(processFrame); return; }
  lastTs=ts;
  resizeCanvas();
  ctx.clearRect(0,0,canvas.width,canvas.height);
  try{
    const res = await poseLandmarker.detectForVideo(videoEl, ts);
    if(res && res.landmarks && res.landmarks.length){
      const lms = res.landmarks[0];
      drawResults(lms);
      computeMetrics(lms, canvas.width, canvas.height);
      updateMetricsUI();
    }
  }catch(e){ setStatus("推論エラー: "+e.message); }
  requestAnimationFrame(processFrame);
}

// Buttons
loadModelBtn.addEventListener("click", ()=>loadModel().catch(e=>setStatus("モデル読込エラー: "+e.message)));
startCamBtn.addEventListener("click", startCamera);
stopCamBtn.addEventListener("click", stopCamera);
playFileBtn?.addEventListener("click", async ()=>{ if(!videoFile.files?.length) alert("動画を選択してください"); });

