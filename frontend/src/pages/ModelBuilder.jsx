import React, { useState, useEffect, useCallback } from 'react';
import { Plus, Trash2, ChevronUp, ChevronDown, Code, Cpu, Check, Copy, Save, Download, Upload, Layers } from 'lucide-react';
import { getLayerCatalog, getModelPresets, validateModel, buildModel, modelToCode, getModelSpecs, saveModelSpec, deleteModelSpec } from '../utils/api';

const CC = {
  linear:'border-blue-500/40', convolution:'border-purple-500/40', pooling:'border-cyan-500/40',
  normalization:'border-yellow-500/40', activation:'border-green-500/40', regularization:'border-red-500/40',
  reshape:'border-orange-500/40', recurrent:'border-pink-500/40', transformer:'border-indigo-500/40', embedding:'border-teal-500/40',
};

function LayerCard({ layer, index, total, onUpdate, onRemove, onMove, shapeInfo }) {
  const [exp, setExp] = useState(false);
  return (
    <div className={`border rounded-lg ${CC[layer._cat]||'border-forge-border'} bg-forge-surface/50`}>
      <div className="flex items-center gap-2 px-3 py-2">
        <span className="text-xs font-mono font-bold text-forge-accent w-5">{index}</span>
        <button onClick={() => setExp(!exp)} className="flex-1 text-left text-sm font-medium">
          {layer.type}
          {shapeInfo?.output_shape && <span className="ml-2 text-xs text-forge-muted font-mono">→[{shapeInfo.output_shape.join(',')}]</span>}
          {shapeInfo?.shape_error && <span className="ml-2 text-xs text-red-400">⚠</span>}
        </button>
        <div className="flex gap-0.5">
          {index>0&&<button onClick={()=>onMove(index,index-1)} className="p-1 text-forge-muted hover:text-white"><ChevronUp className="w-3 h-3"/></button>}
          {index<total-1&&<button onClick={()=>onMove(index,index+1)} className="p-1 text-forge-muted hover:text-white"><ChevronDown className="w-3 h-3"/></button>}
          <button onClick={()=>onRemove(index)} className="p-1 text-forge-muted hover:text-red-400"><Trash2 className="w-3 h-3"/></button>
        </div>
      </div>
      {exp&&(
        <div className="px-3 pb-3 pt-1 border-t border-forge-border/20 grid grid-cols-2 gap-2">
          {Object.entries(layer).filter(([k])=>k!=='type'&&!k.startsWith('_')).map(([k,v])=>(
            <div key={k}>
              <label className="text-xs text-forge-muted">{k}</label>
              {typeof v==='boolean'?(
                <input type="checkbox" checked={v} onChange={e=>onUpdate(index,{...layer,[k]:e.target.checked})} className="accent-forge-accent"/>
              ):(
                <input type={typeof v==='number'?'number':'text'} value={v??''}
                  onChange={e=>{let val=e.target.value;if(typeof v==='number')val=parseFloat(val)||0;onUpdate(index,{...layer,[k]:val});}}
                  className="w-full bg-forge-bg border border-forge-border rounded px-2 py-1 text-xs font-mono focus:border-forge-accent focus:outline-none"/>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function ModelBuilder() {
  const [catalog, setCatalog] = useState({});
  const [presets, setPresets] = useState({});
  const [savedSpecs, setSavedSpecs] = useState([]);
  const [layers, setLayers] = useState([]);
  const [modelName, setModelName] = useState('CustomModel');
  const [inputShape, setInputShape] = useState('1,28,28');
  const [shapesInfo, setShapesInfo] = useState(null);
  const [buildInfo, setBuildInfo] = useState(null);
  const [code, setCode] = useState('');
  const [msg, setMsg] = useState('');
  const [openCat, setOpenCat] = useState(null);
  const [saveName, setSaveName] = useState('');

  const reload = () => {
    getLayerCatalog().then(d=>setCatalog(d.layers||{})).catch(()=>{});
    getModelPresets().then(d=>setPresets(d.presets||{})).catch(()=>{});
    getModelSpecs().then(d=>setSavedSpecs(d.specs||[])).catch(()=>{});
  };
  useEffect(reload,[]);

  const spec = { name: modelName, layers: layers.map(({_cat,...r})=>r), input_shape: inputShape.split(',').map(Number).filter(Boolean) };
  const pShape = inputShape.split(',').map(Number).filter(Boolean);

  const addLayer = useCallback(type=>{
    const reg=catalog[type]; if(!reg) return;
    const nl={type,_cat:reg.category};
    Object.entries(reg.defaults||{}).forEach(([k,v])=>{nl[k]=v;});
    Object.entries(reg.params||{}).forEach(([k,pt])=>{if(!(k in nl)){if(pt==='int')nl[k]=0;else if(pt==='float')nl[k]=0.0;else if(pt==='bool')nl[k]=true;}});
    setLayers(p=>[...p,nl]);
  },[catalog]);

  const updateLayer=(i,l)=>setLayers(p=>p.map((x,j)=>j===i?l:x));
  const removeLayer=i=>setLayers(p=>p.filter((_,j)=>j!==i));
  const moveLayer=(f,t)=>setLayers(p=>{const a=[...p];const[it]=a.splice(f,1);a.splice(t,0,it);return a;});

  const loadPreset=k=>{
    const pr=presets[k]; if(!pr) return;
    setModelName(pr.name||k); setInputShape(pr.input_shape?.join(',')||'1,28,28');
    setLayers(pr.layers.map(l=>({...l,_cat:catalog[l.type]?.category||'linear'})));
    setBuildInfo(null); setShapesInfo(null); setCode('');
  };

  const loadSaved=async(name)=>{
    try{const{spec:s}=await(await fetch(`/api/model_specs/${name}`)).json();
    setModelName(s.name||name); setInputShape((s.input_shape||[1,28,28]).join(','));
    setLayers((s.layers||[]).map(l=>({...l,_cat:catalog[l.type]?.category||'linear'})));
    setSaveName(name); setBuildInfo(null);setShapesInfo(null);setCode('');setMsg(`Loaded: ${name}`);}catch(e){setMsg(e.message);}
  };

  const doSave=async()=>{
    const n=saveName.trim()||modelName.replace(/\s+/g,'_');
    if(!n){setMsg('Enter a name');return;}
    try{await saveModelSpec(n,spec);setSaveName(n);reload();setMsg(`Saved: ${n}`);}catch(e){setMsg(e.message);}
  };

  const doDelete=async(name)=>{
    if(!confirm(`Delete spec "${name}"?`))return;
    try{await deleteModelSpec(name);reload();}catch(e){setMsg(e.message);}
  };

  const doValidate=async()=>{try{const r=await validateModel(spec,pShape);setShapesInfo(r.layers_with_shapes);setMsg(r.errors?.length?`Errors: ${r.errors.join(', ')}`:'Valid!');}catch(e){setMsg(e.message);}};
  const doBuild=async()=>{try{const r=await buildModel(spec);setBuildInfo(r);setMsg(`Built: ${r.total_params_human}`);}catch(e){setMsg(e.message);}};
  const doCode=async()=>{try{const r=await modelToCode(spec);setCode(r.code);}catch(e){setMsg(e.message);}};

  // Export spec as JSON file download
  const doExport = () => {
    const blob = new Blob([JSON.stringify(spec, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${modelName.replace(/\s+/g, '_')}_spec.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Import spec from JSON file
  const doImport = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      try {
        const text = await file.text();
        const imported = JSON.parse(text);
        if (imported.layers) {
          setModelName(imported.name || 'Imported');
          setInputShape((imported.input_shape || [1, 28, 28]).join(','));
          setLayers((imported.layers || []).map(l => ({ ...l, _cat: catalog[l.type]?.category || 'linear' })));
          setBuildInfo(null); setShapesInfo(null); setCode('');
          setMsg(`Imported: ${imported.name || file.name}`);
        } else {
          setMsg('Invalid spec: missing layers array');
        }
      } catch (err) {
        setMsg(`Import error: ${err.message}`);
      }
    };
    input.click();
  };

  const categories=[...new Set(Object.values(catalog).map(c=>c.category))];

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold">Model Builder</h1>
        <div className="flex gap-2 flex-wrap items-center">
          <button onClick={doImport} className="flex items-center gap-1 px-3 py-1.5 bg-forge-surface border border-forge-border rounded text-xs hover:border-forge-accent">
            <Upload className="w-3 h-3"/> Import Spec
          </button>
          <button onClick={doExport} disabled={layers.length===0} className="flex items-center gap-1 px-3 py-1.5 bg-forge-surface border border-forge-border rounded text-xs hover:border-forge-accent disabled:opacity-40">
            <Download className="w-3 h-3"/> Export Spec
          </button>
          <span className="text-forge-border">|</span>
          {Object.keys(presets).map(k=>(
            <button key={k} onClick={()=>loadPreset(k)} className="px-2 py-1 text-xs bg-forge-surface border border-forge-border rounded hover:border-forge-accent font-mono">{k}</button>
          ))}
        </div>
      </div>

      {/* Saved specs bar */}
      {savedSpecs.length>0&&(
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-forge-muted uppercase">Saved:</span>
          {savedSpecs.map(s=>(
            <div key={s.name} className="flex items-center gap-1">
              <button onClick={()=>loadSaved(s.name)} className="px-2 py-1 text-xs bg-forge-accent/10 text-forge-accent border border-forge-accent/30 rounded hover:bg-forge-accent/20 font-mono">{s.name}</button>
              <button onClick={()=>doDelete(s.name)} className="text-forge-muted hover:text-red-400"><Trash2 className="w-3 h-3"/></button>
            </div>
          ))}
        </div>
      )}

      <div className="flex gap-5">
        {/* LEFT: palette */}
        <div className="w-48 shrink-0 space-y-3">
          <div>
            <label className="text-xs text-forge-muted">Name</label>
            <input value={modelName} onChange={e=>setModelName(e.target.value)} className="w-full bg-forge-bg border border-forge-border rounded px-2 py-1.5 text-sm font-mono focus:border-forge-accent focus:outline-none"/>
          </div>
          <div>
            <label className="text-xs text-forge-muted">Input Shape</label>
            <input value={inputShape} onChange={e=>setInputShape(e.target.value)} className="w-full bg-forge-bg border border-forge-border rounded px-2 py-1.5 text-sm font-mono focus:border-forge-accent focus:outline-none"/>
          </div>
          <div className="text-xs text-forge-muted uppercase tracking-wide pt-1">Layers</div>
          {categories.map(cat=>(
            <div key={cat}>
              <button onClick={()=>setOpenCat(openCat===cat?null:cat)} className="w-full text-left text-xs font-medium uppercase tracking-wide text-forge-muted hover:text-forge-accent py-0.5">
                {openCat===cat?'▾':'▸'} {cat}
              </button>
              {openCat===cat&&(
                <div className="space-y-0.5 ml-1">
                  {Object.entries(catalog).filter(([,v])=>v.category===cat).map(([type])=>(
                    <button key={type} onClick={()=>addLayer(type)} className="w-full text-left text-xs font-mono px-2 py-0.5 rounded hover:bg-forge-accent/10 hover:text-forge-accent flex items-center gap-1">
                      <Plus className="w-3 h-3"/> {type}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* CENTER: stack */}
        <div className="flex-1 space-y-3">
          <div className="text-center text-xs font-mono text-forge-muted border border-dashed border-forge-border rounded py-1.5">
            Input [{pShape.join(', ')}]
          </div>

          {layers.length===0?(
            <div className="text-center py-14 text-forge-muted text-sm">Add layers from palette or load a preset / saved spec</div>
          ):(
            <div className="space-y-1.5">
              {layers.map((l,i)=><LayerCard key={i} layer={l} index={i} total={layers.length} onUpdate={updateLayer} onRemove={removeLayer} onMove={moveLayer} shapeInfo={shapesInfo?.[i]}/>)}
            </div>
          )}

          <div className="flex gap-2 flex-wrap pt-2">
            <button onClick={doValidate} className="flex items-center gap-1 px-3 py-1.5 bg-forge-surface border border-forge-border rounded text-xs hover:border-forge-accent"><Check className="w-3 h-3"/> Validate</button>
            <button onClick={doBuild} className="flex items-center gap-1 px-3 py-1.5 bg-forge-accent text-black rounded text-xs font-medium"><Cpu className="w-3 h-3"/> Build</button>
            <button onClick={doCode} className="flex items-center gap-1 px-3 py-1.5 bg-forge-surface border border-forge-border rounded text-xs hover:border-forge-accent"><Code className="w-3 h-3"/> Code</button>
            <div className="flex items-center gap-1">
              <input value={saveName} onChange={e=>setSaveName(e.target.value)} placeholder={modelName} className="w-32 bg-forge-bg border border-forge-border rounded px-2 py-1.5 text-xs font-mono focus:border-forge-accent focus:outline-none"/>
              <button onClick={doSave} className="flex items-center gap-1 px-3 py-1.5 bg-green-600 text-white rounded text-xs font-medium"><Save className="w-3 h-3"/> Save Spec</button>
            </div>
          </div>

          {buildInfo&&(
            <div className="bg-forge-surface border border-forge-border rounded p-3 text-sm flex gap-4">
              <span className="text-forge-accent font-bold">{buildInfo.model_name}</span>
              <span className="text-forge-muted">{buildInfo.total_params_human}</span>
              <span className="text-forge-muted">{buildInfo.num_layers} layers</span>
            </div>
          )}

          {code&&(
            <div className="bg-forge-surface border border-forge-border rounded-lg overflow-hidden">
              <div className="flex items-center justify-between px-4 py-2 border-b border-forge-border">
                <span className="text-xs font-medium">Python Code</span>
                <button onClick={()=>navigator.clipboard.writeText(code)} className="text-xs text-forge-muted hover:text-forge-accent flex items-center gap-1"><Copy className="w-3 h-3"/> Copy</button>
              </div>
              <pre className="px-4 py-3 text-xs font-mono text-forge-text overflow-auto max-h-56">{code}</pre>
            </div>
          )}

          {/* Usage hint */}
          {savedSpecs.length > 0 && (
            <div className="bg-forge-bg border border-forge-border rounded p-3 text-xs text-forge-muted">
              <strong className="text-forge-text">Next steps:</strong> Go to <strong>Configurations</strong> to create a training config that references your saved model spec, then use the <strong>Training</strong> page to start training.
            </div>
          )}

          {msg&&<p className="text-sm text-forge-warning">{msg}</p>}
        </div>
      </div>
    </div>
  );
}
