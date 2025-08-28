// Gait analysis using MediaPipe Tasks Vision (Pose Landmarker)
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

const IDX = {
  nose: 0, l_eye_inner: 1, l_eye: 2, l_eye_outer: 3, r_eye_inner: 4, r_eye: 5, r_eye_outer: 6,
  l_ear: 7, r_ear: 8, l_mouth: 9, r_mouth: 10,
  l_shoulder: 11, r_shoulder: 12, l_elbow: 13, r_elbow: 14, l_wrist: 15, r_wrist: 16,
  l_pinky: 17, r_pinky: 18, l_index: 19, r_index: 20, l_thumb: 21, r_thumb: 22,
  l_hip: 23, r_hip: 24, l_knee: 25, r_knee: 26, l_ankle: 27, r_ankle: 28,
  l_heel: 29, r_heel: 30, l_foot_index: 31, r_foot_index: 32
};

function avg(a){ return a.length ? a.reduce((x,y)=>x+y,0)/a.length : NaN; }

function angleDeg(a,b,c){
  const v1={x:a.x-b.x,y:a.y-b.y}, v2={x:c.x-b.x,y:c.y-b.y};
  const dot=v1.x*v2.x+v1.y*v2.y, m1=Math.hypot(v1.x,v1.y), m2=Math.hypot(v2.x,v2.y);
  if(m1===0||m2===0) return NaN;
  let cos=dot/(m1*m2); cos=Math.max(-1,Math.min(1,cos));
  return Math.acos(cos)*180/Math.PI;
}
function tibiaTiltDeg(ankle,knee){
  const dx=ankle.x-knee.x, dy=ankle.y-knee.y;
  const ang=Math.atan2(dx,Math.abs(dy));
  return ang*180/Math.PI;
}
function toPx(l,w,h){ return {x:l.x*w,y:l.y*h,z:l.z}; }

async function loadModel(){
  statusBox.textContent="モデル読込中...";
  const fr = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
  poseLandmarker = await PoseLandmarker.createFromOptions(fr, {
    baseOptions:{
      modelAssetPath:"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm/pose_landmarker_full.task",
      delegate:"GPU"
    },
    runningMode:"VIDEO",
    numPoses:1
  });
  statusBox.textContent="モデル読込完了";
}

async function startCamera(){
  if(!poseLandmarker) await loadModel();
  stream = await navigator.mediaDevices.getUserMedia({video:{facingMode:"environment"}, audio:false});
  videoEl.srcObject = stream;
  await videoEl.play();
  videoPlaying = true;
  canvas.width = videoEl.videoWidth;
  canvas.height = videoEl.videoHeight;
  statusBox.textContent="カメラ動作中";
  requestAnimationFrame(processFrame);
  startCamBtn.disabled=true; stopCamBtn.disabled=false;
}
function stopCamera(){
  if(stream){ for(const t of stream.getTracks()) t.stop(); }
  videoEl.pause(); videoEl.srcObject=null; videoPlaying=false;
  startCamBtn.disabled=false; stopCamBtn.disabled=true;
  statusBox.textContent="停止";
}
async function playVideoFile(){
  if(!poseLandmarker) await loadModel();
  const file = videoFile.files?.[0]; if(!file){ alert("動画を選択してください"); return; }
  const url = URL.createObjectURL(file);
  videoEl.srcObject = null; videoEl.src = url;
  await videoEl.play();
  videoPlaying=true;
  canvas.width=videoEl.videoWidth; canvas.height=videoEl.videoHeight;
  statusBox.textContent="動画解析中...";
  lastTs=-1;
  requestAnimationFrame(processFrame);
}

function drawResults(lms){
  const du = new DrawingUtils(ctx);
  if(document.getElementById("drawSkeleton").checked){
    du.drawLandmarks(lms,{radius:3});
    du.drawConnectors(lms, PoseLandmarker.POSE_CONNECTIONS);
  }else if(document.getElementById("showKeypoints").checked){
    du.drawLandmarks(lms,{radius:2});
  }
}

function updateMetricsUI(){
  const vL=avg(metricsBuffer.valgusL), vR=avg(metricsBuffer.valgusR), pD=avg(metricsBuffer.pelvicDrop),
        sW=avg(metricsBuffer.stepWidth), tT=avg(metricsBuffer.tibiaTilt);
  valgusL.textContent=isNaN(vL)?"-":vL.toFixed(1);
  valgusR.textContent=isNaN(vR)?"-":vR.toFixed(1);
  pelvicDrop.textContent=isNaN(pD)?"-":pD.toFixed(1);
  stepWidth.textContent=isNaN(sW)?"-":sW.toFixed(2);
  tibiaTilt.textContent=isNaN(tT)?"-":tT.toFixed(1);

  const sideOnly = document.querySelectorAll(".side-only");
  sideOnly.forEach(el=>{ el.style.display = (viewSelect.value==="side") ? "list-item":"none"; });

  function hint(el,val,ok){ if(isNaN(val)){ el.parentElement.style.color=""; return; } el.parentElement.style.color = ok(val)? "":"#b30000"; }
  hint(valgusL,vL,v=>v>=165); hint(valgusR,vR,v=>v>=165);
  hint(pelvicDrop,pD,v=>v<=5); hint(stepWidth,sW,v=>v>=0.3);
  if(viewSelect.value==="side"){ hint(tibiaTilt,tT,v=>v<=8); }
}

function computeMetrics(lms,w,h){
  const L=(i)=>toPx(lms[i],w,h);
  const lh=L(IDX.l_hip), rh=L(IDX.r_hip), lk=L(IDX.l_knee), rk=L(IDX.r_knee), la=L(IDX.l_ankle), ra=L(IDX.r_ankle);
  const l_vis=lms[IDX.l_hip].visibility>0.3 && lms[IDX.l_knee].visibility>0.3 && lms[IDX.l_ankle].visibility>0.3;
  const r_vis=lms[IDX.r_hip].visibility>0.3 && lms[IDX.r_knee].visibility>0.3 && lms[IDX.r_ankle].visibility>0.3;

  const vl=angleDeg(lh,lk,la), vr=angleDeg(rh,rk,ra);
  if(!isNaN(vl)) metricsBuffer.valgusL.push(vl);
  if(!isNaN(vr)) metricsBuffer.valgusR.push(vr);

  if(l_vis && r_vis){
    const hipW=Math.max(1, Math.hypot(lh.x-rh.x, lh.y-rh.y));
    const dy=(lh.y-rh.y);
    const angle=Math.atan2(Math.abs(dy), hipW)*180/Math.PI;
    metricsBuffer.pelvicDrop.push(angle);
    const sw=Math.abs(la.x - ra.x)/hipW;
    metricsBuffer.stepWidth.push(sw);
  }

  const tibL=tibiaTiltDeg(la,lk), tibR=tibiaTiltDeg(ra,rk);
  const use = la.y>ra.y ? Math.abs(tibL) : Math.abs(tibR);
  if(!isNaN(use)) metricsBuffer.tibiaTilt.push(use);

  for(const k of Object.keys(metricsBuffer)){
    if(metricsBuffer[k].length>MAX_BUF) metricsBuffer[k].shift();
  }
}

async function processFrame(ts){
  if(!videoPlaying) return;
  if(lastTs===ts){ requestAnimationFrame(processFrame); return; }
  lastTs=ts;

  const w=videoEl.videoWidth, h=videoEl.videoHeight;
  canvas.width=w; canvas.height=h;
  ctx.clearRect(0,0,w,h);

  try{
    const res = await poseLandmarker.detectForVideo(videoEl, ts);
    if(res && res.landmarks && res.landmarks.length){
      const lms = res.landmarks[0];
      drawResults(lms);
      computeMetrics(lms, w, h);
      updateMetricsUI();
    }
  }catch(e){
    console.error(e);
    statusBox.textContent="推論エラー: "+e.message;
  }finally{
    requestAnimationFrame(processFrame);
  }
}

document.getElementById("loadModelBtn").addEventListener("click", loadModel);
document.getElementById("startCamBtn").addEventListener("click", startCamera);
document.getElementById("stopCamBtn").addEventListener("click", stopCamera);
document.getElementById("playFileBtn").addEventListener("click", playVideoFile);

updateMetricsUI();
