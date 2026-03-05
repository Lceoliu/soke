import os
import sys
from pathlib import Path

# --- 1. CRITICAL: MUST SET ENVIRONMENT BEFORE ANY IMPORTS ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

BACKEND_PREF_FILE = Path(ROOT_DIR) / "dashboard" / ".backend_pref"
def get_backend_pref():
    if BACKEND_PREF_FILE.exists():
        return BACKEND_PREF_FILE.read_text().strip()
    return "osmesa" # Default to osmesa for Docker stability

current_backend = get_backend_pref()
os.environ["PYOPENGL_PLATFORM"] = current_backend

if current_backend == "osmesa":
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    os.environ["GALLIUM_DRIVER"] = "softpipe"

# --- 2. NOW IMPORT EVERYTHING ELSE ---
import streamlit as st
from dashboard.utils.exp_manager import list_experiments, run_generate_report
from dashboard.utils.visualizer import StreamVisualizer, load_pose_dir
from dashboard.utils.dataset_manager import DatasetManager

st.set_page_config(page_title="SOKE Management Dashboard", layout="wide")

if 'selected_poses' not in st.session_state:
    st.session_state['selected_poses'] = []

st.title("🚀 SOKE Management Dashboard")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Navigation")
menu = st.sidebar.radio("Go to", ["Experiment Explorer", "Visualizer Studio"])

st.sidebar.markdown("---")
st.sidebar.header("Settings")
backend_options = ["osmesa", "egl"]
selected_backend = st.sidebar.selectbox(
    "Rendering Backend", 
    backend_options, 
    index=backend_options.index(current_backend),
    help="osmesa: Software rendering. egl: GPU rendering."
)

if selected_backend != current_backend:
    BACKEND_PREF_FILE.write_text(selected_backend)
    st.sidebar.success(f"Backend set to **{selected_backend}**")
    st.sidebar.warning("⚠️ RESTART REQUIRED: Restart streamlit now.")

# --- PAGES ---
if menu == "Experiment Explorer":
    st.header("🧪 Experiment Explorer")
    exps = list_experiments()
    
    if not exps:
        st.info("No valid experiments found in experiments/mgpt.")
    else:
        for exp in exps:
            with st.expander(f"📁 {exp['name']}"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write(f"**Path:** `{exp['path']}`")
                    st.write(f"**Config:** `{exp['config_path']}`")
                    
                    if st.button("📊 Generate Report", key=f"rep_{exp['name']}"):
                        with st.spinner("Generating..."):
                            ok, msg = run_generate_report(exp)
                            if ok: st.success("Report generated!")
                            else: st.error(msg)
                    
                    report_dir = Path(exp['path']) / "auto_reports" / "rvq_stage1"
                    if report_dir.exists():
                        images = sorted(list(report_dir.glob("*.png")))
                        for img in images: st.image(str(img), caption=img.name)
                with col2:
                    if exp['config']: st.json(exp['config'])

elif menu == "Visualizer Studio":
    st.header("🎬 Visualizer Studio")
    dm = DatasetManager()
    datasets = dm.list_datasets()
    
    left_col, right_col = st.columns([1, 2])
    
    with left_col:
        st.subheader("📂 Dataset Browser")
        if not datasets:
            st.warning("No datasets found in data/ folder.")
        else:
            ds_names = [d['name'] for d in datasets]
            selected_ds_name = st.selectbox("Select Dataset", ds_names)
            selected_ds = next(d for d in datasets if d['name'] == selected_ds_name)
            
            # --- RESTORED STATISTICS ---
            stats = dm.get_statistics(selected_ds['path'])
            st.write(f"**Total Poses:** {stats['total_poses']}")
            if stats['splits']:
                cols = st.columns(len(stats['splits']))
                for i, (split, count) in enumerate(stats['splits'].items()):
                    cols[i].metric(split.capitalize(), count)
            
            st.markdown("---")
            # --- RESTORED SPLIT SEARCH ---
            search_split = st.selectbox("Search in Split", ["all", "train", "val", "test"], index=0)
            pose_names = dm.get_pose_list(selected_ds['path'], split=search_split)
            target_pose = st.selectbox(f"Search Pose ({len(pose_names)})", [""] + pose_names)
            
            if target_pose and st.button("➕ Add to Queue"):
                pose_id = f"{selected_ds_name}_{target_pose}"
                if pose_id not in [p['id'] for p in st.session_state['selected_poses']]:
                    st.session_state['selected_poses'].append({
                        "id": pose_id, "name": target_pose, 
                        "ds_path": selected_ds['path'], "ds_name": selected_ds_name
                    })
                    st.rerun()

    with right_col:
        st.subheader("🛠 Visualization Queue")
        if not st.session_state['selected_poses']:
            st.info("Queue is empty.")
        else:
            options = [p['id'] for p in st.session_state['selected_poses']]
            selected_ids = st.multiselect("Queue Management", options=options, default=options)
            
            if len(selected_ids) != len(st.session_state['selected_poses']):
                st.session_state['selected_poses'] = [p for p in st.session_state['selected_poses'] if p['id'] in selected_ids]
                st.rerun()

            if st.button("🚀 Run Visualization"):
                st.write("---")
                primary_pose = st.session_state['selected_poses'][0]
                img_placeholder = st.empty()
                progress_bar = st.progress(0)
                
                try:
                    viz = StreamVisualizer(width=512, height=512)
                    p_path = dm.resolve_pose_path(primary_pose['ds_path'], primary_pose['name'])
                    if p_path:
                        root, body, lhand, rhand, jaw, shape, expr = load_pose_dir(p_path)
                        vertices_seq = viz.smplx_to_vertices(root, body, lhand, rhand, jaw, shape, expr)
                        
                        for i in range(len(vertices_seq)):
                            frame = viz.render_frame(vertices_seq[i])
                            img_placeholder.image(frame, channels="RGB", caption=f"Rendering: {primary_pose['name']} ({i}/{len(vertices_seq)})")
                            progress_bar.progress((i + 1) / len(vertices_seq))
                        st.success(f"Finished: {primary_pose['name']}")
                    else: st.error("Path unresolved.")
                    viz.cleanup()
                except Exception as e:
                    st.error(f"Render Error: {e}")

st.sidebar.markdown("---")
st.sidebar.info(f"Backend: {current_backend}")
