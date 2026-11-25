function makeTwin(P){
const N=P.param_names, idx={}; N.forEach((n,i)=>idx[n]=i);
const FAM=["fast","milk","starch","legume","veg"], DT=5;
const E=(x,n)=>{if(x<=0)return 0;const e=Math.exp(-x);return n===2?1-(1+x)*e:1-(1+x+0.5*x*x)*e};
function curve(th,ev,g0,mod,steps){
  const p=k=>th[idx[k]], shift=720, n=(steps*5+shift)/DT+145, inow=shift/DT;
  const ins=new Float64Array(n), ra=new Float64Array(n);
  for(const e of ev){const bt=e.t+shift;
    if(e.u>0){const j0=Math.floor(bt/DT),ti=p("ti");for(let j=Math.max(j0,0);j<Math.min(n,j0+120);j++){const t=j*DT-bt;ins[j]+=e.u*(E((t+DT)/ti,3)-E(t/ti,3))}}
    if(e.food){const f=e.food;let slow=1+p("fat_slow")*f.fat/30;if(f.low)slow*=p("lowfast");const j0=Math.floor(bt/DT);
      for(let j=Math.max(j0,0);j<Math.min(n,j0+144);j++){const t=j*DT-bt;let s=0;
        for(let k=0;k<5;k++)if(f.fam[k]>0){const tf=p("tau_"+FAM[k])*slow;s+=f.fam[k]*p("cf_"+FAM[k])*(E((t+DT)/tf,2)-E(t/tf,2))}
        if(f.fpu>0){const tp=p("tau_fpu");s+=f.fpu*p("cf_fpu")*(E((t+DT)/tp,2)-E(t/tp,2))}
        ra[j]+=s}}}
  const out=[];let G=g0;
  for(let s=0;s<steps;s++){const j=inow+s,m=((mod+(j-inow)*DT)%1440+1440)%1440,ts=Math.sin(2*Math.PI*m/1440),tc=Math.cos(2*Math.PI*m/1440);
    const sif=Math.max(0.2,1+p("si_s")*ts+p("si_c")*tc),drift=p("dr0")+p("dr_s")*ts+p("dr_c")*tc;
    let f=1+p("beta")*(G-120)/120;if(f<0.3)f=0.3;const low=G<85?p("c_low")*(85-G)*DT:0;
    G=G-p("k")*(G-p("Gb"))*DT+ra[j]-p("SI")*ins[j]*sif*f+drift*DT/60+low;if(G<20)G=20;out.push(G)}
  return out}
return {curve, idx}}
if(typeof module!=="undefined")module.exports=makeTwin;
