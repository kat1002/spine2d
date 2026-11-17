use egui_wgpu::Renderer as EguiRenderer;
use egui_wgpu::RendererOptions as EguiRendererOptions;
use egui_wgpu::ScreenDescriptor as EguiScreenDescriptor;
use spine2d::{AnimationState, AnimationStateData, Atlas, DrawList, Skeleton, SkeletonData};
use spine2d_wgpu::{
    HashMapTextureProvider, SpineRenderer, create_sampler_for_atlas_page,
    create_texture_bind_group, present_surface_texture, wgpu,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

fn sanitize_example_arg(raw: &str) -> String {
    let trimmed = raw.trim();
    let mut out = String::new();
    for ch in trimmed.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch.to_ascii_lowercase());
        } else {
            break;
        }
    }
    if out.is_empty() {
        trimmed.to_string()
    } else {
        out
    }
}

fn upstream_examples_root() -> Result<PathBuf, String> {
    if let Ok(dir) = std::env::var("SPINE2D_UPSTREAM_EXAMPLES_DIR") {
        let p = PathBuf::from(dir);
        if p.is_dir() {
            return Ok(p);
        }
    }

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        root.join("../assets/spine-runtimes/examples"),
        root.join("../third_party/spine-runtimes/examples"),
        root.join("../.cache/spine-runtimes/examples"),
    ];
    for p in candidates {
        if p.is_dir() {
            return Ok(p);
        }
    }

    Err("Upstream Spine examples not found.\n\
Run `python3 ./scripts/fetch_spine_runtimes_examples.py --mode export --scope tests` or set \
`SPINE2D_UPSTREAM_EXAMPLES_DIR` to `<spine-runtimes>/examples`."
        .to_string())
}

fn available_examples() -> Vec<String> {
    let Ok(root) = upstream_examples_root() else {
        return Vec::new();
    };
    let Ok(rd) = std::fs::read_dir(&root) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for entry in rd.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if path.join("export").is_dir() {
            out.push(name.to_string());
        }
    }
    out.sort();
    out
}

fn first_file_with_extension(dir: &Path, ext: &str) -> Option<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return None;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_file() && path.extension().is_some_and(|e| e == ext) {
            return Some(path);
        }
    }
    None
}

fn preferred_existing(paths: &[PathBuf]) -> Option<PathBuf> {
    paths.iter().find(|p| p.is_file()).cloned()
}

fn pick_export_files(example: &str, prefer_pma_atlas: bool) -> Result<(PathBuf, PathBuf), String> {
    let examples = upstream_examples_root()?;
    let export_dir = examples.join(example).join("export");
    if !export_dir.is_dir() {
        return Err(format!(
            "Missing export dir: {export_dir:?}\n\
Example name should match a directory under `<spine-runtimes>/examples/` (e.g. `spineboy`, `alien`)."
        ));
    }

    let json = preferred_existing(&[
        export_dir.join(format!("{example}-pro.json")),
        export_dir.join(format!("{example}-ess.json")),
        export_dir.join(format!("{example}.json")),
    ])
    .or_else(|| first_file_with_extension(&export_dir, "json"))
    .ok_or_else(|| format!("No json found in {export_dir:?}"))?;

    let atlas_candidates = if prefer_pma_atlas {
        vec![
            export_dir.join(format!("{example}-pma.atlas")),
            export_dir.join(format!("{example}-run.atlas")),
            export_dir.join(format!("{example}.atlas")),
        ]
    } else {
        vec![
            export_dir.join(format!("{example}.atlas")),
            export_dir.join(format!("{example}-run.atlas")),
            export_dir.join(format!("{example}-pma.atlas")),
        ]
    };

    let atlas = preferred_existing(&atlas_candidates)
        .or_else(|| first_file_with_extension(&export_dir, "atlas"))
        .ok_or_else(|| {
            format!(
                "No atlas found in {export_dir:?}\n\
You probably imported only JSON. Re-import with:\n\
  `python3 ./scripts/fetch_spine_runtimes_examples.py --mode export --scope tests`"
            )
        })?;

    Ok((json, atlas))
}

fn load_png_rgba8(path: &Path) -> Result<(u32, u32, Vec<u8>), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read png {path:?}: {e}"))?;
    let img = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
        .map_err(|e| format!("decode png {path:?}: {e}"))?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    Ok((w, h, rgba.into_raw()))
}

/// Copies the sprite file next to the atlas and registers the `__custom_slot_*__` region.
fn ensure_atlas_region(
    atlas_str: &mut String,
    export_dir: &Path,
    slot_name: &str,
    cfg: &CustomSpriteConfig,
) -> (String, u32, u32) {
    let src = PathBuf::from(&cfg.source_path);
    let file_name = src
        .file_name()
        .and_then(|s| s.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| format!("custom_{slot_name}.png"));
    let dst = export_dir.join(&file_name);
    if src != dst {
        let _ = std::fs::copy(&src, &dst);
    }
    let region_name = format!("__custom_slot_{slot_name}__");
    let w = cfg.width.round() as u32;
    let h = cfg.height.round() as u32;
    if !atlas_str.contains(&region_name) {
        atlas_str.push_str(&format!(
            "\n{file_name}\nsize:{w},{h}\nfilter:Linear,Linear\npma:false\n{region_name}\nbounds:0,0,{w},{h}\n"
        ));
    }
    (region_name, w, h)
}

fn json_skin_set_custom_region(
    skin: &mut serde_json::Value,
    slot_name: &str,
    region_name: &str,
    cfg: &CustomSpriteConfig,
    w: u32,
    h: u32,
) {
    let Some(atts) = skin.get_mut("attachments").and_then(|v| v.as_object_mut()) else {
        return;
    };
    let slot_entry = atts
        .entry(slot_name.to_string())
        .or_insert_with(|| serde_json::json!({}));
    if let Some(slot_obj) = slot_entry.as_object_mut() {
        slot_obj.insert(
            region_name.to_string(),
            serde_json::json!({
                "path": region_name,
                "x": cfg.x,
                "y": cfg.y,
                "rotation": cfg.rotation,
                "scaleX": cfg.scale_x,
                "scaleY": cfg.scale_y,
                "width": w,
                "height": h
            }),
        );
    }
}

/// Copies attachments referenced by cross-skin overrides into the skin named `target`.
fn copy_cross_skin_overrides(
    skins: &mut [serde_json::Value],
    overrides: &HashMap<String, SlotOverride>,
    target: &str,
) {
    let mut copied: Vec<(String, String, serde_json::Value)> = Vec::new();
    for (slot_name, ov) in overrides {
        if let SlotOverride::Attachment { skin, attachment } = ov
            && skin != target
        {
            for s in skins.iter() {
                if s.get("name").and_then(|v| v.as_str()) == Some(skin.as_str())
                    && let Some(att_val) = s
                        .get("attachments")
                        .and_then(|a| a.get(slot_name))
                        .and_then(|sl| sl.get(attachment))
                {
                    copied.push((slot_name.clone(), attachment.clone(), att_val.clone()));
                }
            }
        }
    }
    for s in skins.iter_mut() {
        if s.get("name").and_then(|v| v.as_str()) == Some(target)
            && let Some(atts) = s.get_mut("attachments").and_then(|v| v.as_object_mut())
        {
            for (slot_name, att_name, val) in &copied {
                let slot_entry = atts
                    .entry(slot_name.clone())
                    .or_insert_with(|| serde_json::json!({}));
                if let Some(slot_obj) = slot_entry.as_object_mut() {
                    slot_obj.insert(att_name.clone(), val.clone());
                }
            }
        }
    }
}

const SKIN_NONE_SENTINEL: &str = "__none__";

#[derive(Clone, Debug)]
struct UiSelection {
    example: String,
    animation: String,
    skin: String,
    prefer_pma_atlas: bool,
    speed: f32,
    margin: f32,
}

#[derive(Copy, Clone, Debug)]
struct Bounds2 {
    min: [f32; 2],
    max: [f32; 2],
}

fn bounds_from_draw_list(draw_list: &DrawList) -> Option<Bounds2> {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for v in &draw_list.vertices {
        min_x = min_x.min(v.position[0]);
        min_y = min_y.min(v.position[1]);
        max_x = max_x.max(v.position[0]);
        max_y = max_y.max(v.position[1]);
    }

    if min_x.is_finite() && min_y.is_finite() && max_x.is_finite() && max_y.is_finite() {
        Some(Bounds2 {
            min: [min_x, min_y],
            max: [max_x, max_y],
        })
    } else {
        None
    }
}

fn clip_from_world_fit_bounds(
    bounds: Bounds2,
    viewport_width: u32,
    viewport_height: u32,
    margin: f32,
) -> [[f32; 4]; 4] {
    let vw = viewport_width.max(1) as f32;
    let vh = viewport_height.max(1) as f32;
    let margin = margin.clamp(0.1, 1.0);

    let world_w = (bounds.max[0] - bounds.min[0]).abs().max(1.0e-3);
    let world_h = (bounds.max[1] - bounds.min[1]).abs().max(1.0e-3);
    let cx = 0.5 * (bounds.min[0] + bounds.max[0]);
    let cy = 0.5 * (bounds.min[1] + bounds.max[1]);

    // Keep world units isotropic in pixels, then convert to clip.
    let scale_px = (vw * margin / world_w).min(vh * margin / world_h);
    let sx = 2.0 * scale_px / vw;
    let sy = 2.0 * scale_px / vh;

    // WGSL matrices are column-major; translation lives in the last column.
    [
        [sx, 0.0, 0.0, 0.0],
        [0.0, sy, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-cx * sx, -cy * sy, 0.0, 1.0],
    ]
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum SlotOverride {
    Hidden,
    Attachment { skin: String, attachment: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StudioTab {
    CurrentSprites,
    WardrobeCustom,
    BoneStudio,
    SkinBuilder,
    LayerOrder,
    ImportExport,
}

#[derive(Clone, Copy, Debug)]
struct PartSpriteTransform {
    x: f32,
    y: f32,
    rotation: f32,
    scale_x: f32,
    scale_y: f32,
    alpha: f32,
}

impl Default for PartSpriteTransform {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            rotation: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
            alpha: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BoneOffset {
    dx: f32,
    dy: f32,
    drot: f32,
    sx: f32,
    sy: f32,
    length: f32,
}

#[derive(Clone, Debug)]
struct CustomSpriteConfig {
    source_path: String,
    x: f32,
    y: f32,
    rotation: f32,
    scale_x: f32,
    scale_y: f32,
    width: f32,
    height: f32,
}

/// A skin built from the current outfit ("Skin Builder"). Kept so it can be re-registered after a
/// reload and baked into the exported JSON.
#[derive(Clone, Debug)]
struct CustomSkinRecord {
    name: String,
    base_skin: String,
    skin: spine2d::SkinData,
    overrides: HashMap<String, SlotOverride>,
    sprites: HashMap<String, CustomSpriteConfig>,
}

const WARDROBE_PRESETS: &[(&str, &str, &str, bool)] = &[
    ("Replace Body / Top", "body", "body", true),
    ("Replace L Hand", "l_hand_C2", "l_hand_C2", true),
    ("Replace R Hand", "r_hand_C2", "r_hand_C2", true),
    ("Replace R Skirt", "body", "r_skirt", true),
    ("Replace L Skirt", "body", "l_skirt", true),
    ("Hat / Hair", "head", "hat", true),
    ("Add L Hand Prop", "l_hand_C2", "custom_l_hand", true),
    ("Add R Hand Prop", "r_hand_C2", "custom_r_hand", true),
    ("Replace L Leg/Shoe", "l_leg", "l_leg", true),
    ("Replace R Leg/Shoe", "r_leg", "r_leg", true),
    ("Wings / Back", "body", "custom_wings", false),
];

const SQUISHY_SKIN_DIR: &str =
    r"D:\Phuc\Projects\squishy_pa\SquishyDumpling\Assets\Resources\Animation\Skin";

/// Lowercase alphanumeric character name derived from a folder / file path.
fn character_name_from_path(path: &Path) -> String {
    let raw = if path.is_dir() {
        path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string()
    } else {
        let file_stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let file_stem = file_stem.split('.').next().unwrap_or(file_stem);
        let parent_name = path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        // If parent has a distinctive name (e.g. folder MiraSanta with mira.json), prefer parent
        if (file_stem.eq_ignore_ascii_case("skeleton") || file_stem.eq_ignore_ascii_case("export"))
            && !parent_name.is_empty()
        {
            parent_name.to_string()
        } else if !parent_name.is_empty()
            && !parent_name.eq_ignore_ascii_case("export")
            && !parent_name.eq_ignore_ascii_case(file_stem)
        {
            parent_name.to_string()
        } else {
            file_stem.to_string()
        }
    };
    raw.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

fn squishy_characters() -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(SQUISHY_SKIN_DIR)
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.path().is_dir())
                .filter_map(|e| e.file_name().to_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out
}

/// Runs `scripts/import_character.py <path>` and returns the imported example name.
fn run_import_character(path: &Path) -> Result<String, String> {
    let name = character_name_from_path(path);
    if name.is_empty() {
        return Err(format!("Cannot derive character name from {path:?}"));
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let script = root.join("scripts").join("import_character.py");
    if !script.is_file() {
        return Err(format!("Importer script not found: {script:?}"));
    }
    let dest_root = upstream_examples_root().ok();
    let mut last_err = String::new();
    for exe in ["python", "py", "python3"] {
        let mut cmd = std::process::Command::new(exe);
        cmd.arg(&script).arg(path).arg("--name").arg(&name);
        if let Some(dest) = dest_root.as_ref() {
            cmd.arg("--dest-root").arg(dest);
        }
        cmd.current_dir(&root);
        match cmd.output() {
            Ok(out) if out.status.success() => return Ok(name),
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                let stdout = String::from_utf8_lossy(&out.stdout);
                return Err(format!(
                    "import_character.py failed ({}): {}{}",
                    out.status,
                    stderr.trim(),
                    stdout.trim()
                ));
            }
            Err(e) => last_err = format!("{exe}: {e}"),
        }
    }
    Err(format!("Could not launch python ({last_err})"))
}

impl Default for BoneOffset {
    fn default() -> Self {
        Self {
            dx: 0.0,
            dy: 0.0,
            drot: 0.0,
            sx: 1.0,
            sy: 1.0,
            length: 0.0,
        }
    }
}

impl BoneOffset {
    fn is_identity(&self) -> bool {
        self.dx.abs() < 1e-4
            && self.dy.abs() < 1e-4
            && self.drot.abs() < 1e-4
            && (self.sx - 1.0).abs() < 1e-4
            && (self.sy - 1.0).abs() < 1e-4
            && self.length.abs() < 1e-4
    }
}

struct App {
    example: String,
    animation: Option<String>,
    speed: f32,
    fit_margin: f32,
    prefer_pma_atlas: bool,
    json_path: PathBuf,
    atlas_path: PathBuf,
    initial_bounds: Option<Bounds2>,

    window: Option<Arc<Window>>,
    surface: Option<wgpu::Surface<'static>>,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    config: Option<wgpu::SurfaceConfiguration>,
    renderer: Option<SpineRenderer>,
    textures: HashMapTextureProvider,

    atlas: Option<Atlas>,
    data: Option<Arc<SkeletonData>>,
    skeleton: Option<Skeleton>,
    state: Option<AnimationState>,

    draw_list: spine2d::DrawList,
    last_frame: Option<Instant>,
    debug_time: bool,
    debug_accum: f32,
    debug_frames: u32,

    egui_ctx: egui::Context,
    egui_state: Option<egui_winit::State>,
    egui_renderer: Option<EguiRenderer>,

    examples: Vec<String>,
    ui: UiSelection,
    pending_reload: Option<UiSelection>,
    paused: bool,
    last_error: Option<String>,
    needs_fit: bool,
    active_tab: StudioTab,
    selected_current_slot: String,
    current_slot_filter: String,
    part_transforms: HashMap<String, PartSpriteTransform>,
    slot_overrides: HashMap<String, SlotOverride>,
    slot_filter: String,
    slot_bone_overrides: HashMap<String, usize>,
    bone_offsets: HashMap<String, BoneOffset>,
    bone_filter: String,
    selected_bone: String,
    created_bones: Vec<(String, String)>,
    bone_parent_overrides: HashMap<String, String>,
    new_bone_name: String,
    new_bone_parent: String,
    custom_draw_order: Option<Vec<usize>>,
    draw_order_filter: String,
    show_bone_gizmos: bool,
    custom_sprite_slot: String,
    custom_sprite_path: String,
    custom_sprites: HashMap<String, CustomSpriteConfig>,
    pending_custom_png: Option<(String, String)>,
    save_status: Option<String>,
    new_slot_name: String,
    new_slot_bone: String,
    created_slots: Vec<(String, String)>,
    new_slot_layer_front: bool,
    selected_part: String,
    new_skin_name: String,
    custom_skins: Vec<CustomSkinRecord>,
    import_status: Option<String>,
}

impl App {
    fn new(
        example: String,
        animation: Option<String>,
        speed: f32,
        fit_margin: f32,
        prefer_pma_atlas: bool,
        json_path: PathBuf,
        atlas_path: PathBuf,
    ) -> Self {
        let egui_ctx = egui::Context::default();
        let ui_animation = animation.clone().unwrap_or_default();
        Self {
            example: example.clone(),
            animation,
            speed: if speed.is_finite() { speed } else { 1.0 },
            fit_margin: if fit_margin.is_finite() {
                fit_margin.clamp(0.1, 1.0)
            } else {
                0.9
            },
            prefer_pma_atlas,
            json_path,
            atlas_path,
            initial_bounds: None,
            window: None,
            surface: None,
            device: None,
            queue: None,
            config: None,
            renderer: None,
            textures: HashMapTextureProvider {
                bind_groups: HashMap::new(),
            },
            atlas: None,
            data: None,
            skeleton: None,
            state: None,
            draw_list: spine2d::DrawList::default(),
            last_frame: None,
            debug_time: std::env::var("SPINE2D_DEBUG_TIME").is_ok(),
            debug_accum: 0.0,
            debug_frames: 0,

            egui_ctx: egui_ctx.clone(),
            egui_state: None,
            egui_renderer: None,

            examples: available_examples(),
            ui: UiSelection {
                example,
                animation: ui_animation,
                skin: String::new(),
                prefer_pma_atlas,
                speed,
                margin: fit_margin,
            },
            pending_reload: None,
            paused: false,
            last_error: None,
            needs_fit: true,
            active_tab: StudioTab::CurrentSprites,
            selected_current_slot: "body".to_string(),
            current_slot_filter: String::new(),
            part_transforms: HashMap::new(),
            slot_overrides: HashMap::new(),
            slot_filter: String::new(),
            slot_bone_overrides: HashMap::new(),
            bone_offsets: HashMap::new(),
            bone_filter: String::new(),
            selected_bone: String::new(),
            created_bones: Vec::new(),
            bone_parent_overrides: HashMap::new(),
            new_bone_name: "custom_bone".to_string(),
            new_bone_parent: "root".to_string(),
            custom_draw_order: None,
            draw_order_filter: String::new(),
            show_bone_gizmos: false,
            custom_sprite_slot: "hat".to_string(),
            custom_sprite_path: String::new(),
            custom_sprites: HashMap::new(),
            pending_custom_png: None,
            save_status: None,
            new_slot_name: "custom_hat".to_string(),
            new_slot_bone: "head".to_string(),
            created_slots: Vec::new(),
            new_slot_layer_front: true,
            selected_part: String::new(),
            new_skin_name: "Custom_Outfit_1".to_string(),
            custom_skins: Vec::new(),
            import_status: None,
        }
    }

    /// Clears every per-character edit (used when switching to another character).
    fn clear_character_edits(&mut self) {
        self.ui.animation.clear();
        self.ui.skin.clear(); // "(auto)"
        self.slot_overrides.clear();
        self.part_transforms.clear();
        self.slot_bone_overrides.clear();
        self.bone_offsets.clear();
        self.selected_bone.clear();
        self.created_bones.clear();
        self.bone_parent_overrides.clear();
        self.custom_draw_order = None;
        self.custom_sprites.clear();
        self.created_slots.clear();
        self.custom_skins.clear();
        self.selected_part.clear();
    }

    /// Imports `path` through `scripts/import_character.py` and loads the result.
    /// Safely copies all files into the example directory so the original external source is never modified.
    fn import_and_load(&mut self, path: &Path) {
        match run_import_character(path) {
            Ok(name) => self.load_imported_example(
                &name,
                format!("✅ Successfully imported '{name}' into examples (original source files preserved untouched)"),
            ),
            Err(e) => {
                self.import_status = None;
                self.last_error = Some(e);
            }
        }
    }

    /// Opens a native file dialog to select any Spine export file (.json, .skel, .skel.bytes, .atlas).
    /// Safely copies the character into the example directory without modifying the original source!
    fn pick_and_import_file(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .set_title("Select Spine Export File (.json, .skel, .skel.bytes, .atlas)")
            .add_filter("Spine Export Files", &["json", "skel", "bytes", "atlas", "txt"])
            .pick_file()
        {
            self.import_and_load(&path);
        }
    }

    /// Opens a native folder dialog to select any Spine character folder.
    /// Safely copies the character into the example directory without modifying the original source!
    fn pick_and_import_folder(&mut self) {
        if let Some(folder) = rfd::FileDialog::new()
            .set_title("Select Spine Character Export Folder")
            .pick_folder()
        {
            self.import_and_load(&folder);
        }
    }

    fn load_imported_example(&mut self, name: &str, status: String) {
        self.examples = available_examples();
        self.clear_character_edits();
        self.ui.example = name.to_string();
        self.import_status = Some(status);
        self.last_error = None;
        self.queue_reload();
    }

    /// Moves `slot_name` to the front (top) or back (behind the character) of the draw order.
    fn set_slot_layer(&mut self, slot_name: &str, front: bool) {
        let Some(data) = self.data.as_ref() else {
            return;
        };
        let Some(slot_idx) = data.find_slot(slot_name).map(|s| s.get_index()) else {
            return;
        };
        let count = data.get_slots().len();
        let mut order = self
            .custom_draw_order
            .clone()
            .or_else(|| self.skeleton.as_ref().map(|sk| sk.get_draw_order().to_vec()))
            .unwrap_or_default();
        if order.len() != count {
            order = (0..count).collect();
        }
        order.retain(|&i| i != slot_idx);
        if front {
            order.push(slot_idx);
        } else {
            order.insert(0, slot_idx);
        }
        self.custom_draw_order = Some(order);
    }

    /// Removes a custom PNG part (attachment, texture and edits) from `slot_name`.
    fn remove_custom_part(&mut self, slot_name: &str) {
        let tex_key = format!("__custom_slot_{slot_name}__");
        self.custom_sprites.remove(slot_name);
        if let Some(SlotOverride::Attachment { attachment, .. }) =
            self.slot_overrides.get(slot_name)
            && attachment == &tex_key
        {
            self.slot_overrides.remove(slot_name);
        }
        self.textures.bind_groups.remove(&tex_key);
        if let Some(data_arc) = self.data.as_mut()
            && let Some(slot_idx) = data_arc.find_slot(slot_name).map(|s| s.get_index())
        {
            let data = Arc::make_mut(data_arc);
            for skin in data.get_skins_mut().iter_mut() {
                skin.remove_attachment(slot_idx, &tex_key);
            }
            if let Some(sk) = self.skeleton.as_mut() {
                sk.set_data(data_arc.clone());
            }
        }
        if self.selected_part == slot_name {
            self.selected_part.clear();
        }
    }

    /// Bakes the current outfit into a new named skin, registers and activates it.
    fn save_current_outfit_as_skin(&mut self, raw_name: &str) -> Result<String, String> {
        let name = raw_name.trim().to_string();
        if name.is_empty() {
            return Err("Skin name cannot be empty".to_string());
        }
        if name == "default" {
            return Err("'default' is reserved; pick another skin name".to_string());
        }
        let Some(data_arc) = self.data.as_mut() else {
            return Err("Skeleton data not loaded".to_string());
        };

        let active_name = self
            .skeleton
            .as_ref()
            .and_then(|sk| sk.get_skin().map(|s| s.get_name().to_string()))
            .unwrap_or_default();

        let mut new_skin = spine2d::SkinData::new(name.clone());
        if let Some(active) = data_arc.find_skin(&active_name) {
            new_skin.copy_skin(active);
        }

        // Cross-skin overrides + custom PNG parts.
        for (slot_name, ov) in &self.slot_overrides {
            let SlotOverride::Attachment { skin, attachment } = ov else {
                continue;
            };
            let Some(slot_idx) = data_arc.find_slot(slot_name).map(|s| s.get_index()) else {
                continue;
            };
            if let Some(att) = data_arc
                .find_skin(skin)
                .and_then(|s| s.get_attachment(slot_idx, attachment))
            {
                new_skin.set_attachment(slot_idx, attachment.clone(), att.clone());
            }
        }
        for slot_name in self.custom_sprites.keys() {
            let tex_key = format!("__custom_slot_{slot_name}__");
            let Some(slot_idx) = data_arc.find_slot(slot_name).map(|s| s.get_index()) else {
                continue;
            };
            if let Some(att) = data_arc
                .find_skin("default")
                .and_then(|s| s.get_attachment(slot_idx, &tex_key))
            {
                new_skin.set_attachment(slot_idx, tex_key, att.clone());
            }
        }

        let record = CustomSkinRecord {
            name: name.clone(),
            base_skin: active_name,
            skin: new_skin.clone(),
            overrides: self.slot_overrides.clone(),
            sprites: self.custom_sprites.clone(),
        };
        Arc::make_mut(data_arc).add_skin(new_skin);
        if let Some(existing) = self.custom_skins.iter_mut().find(|r| r.name == name) {
            *existing = record;
        } else {
            self.custom_skins.push(record);
        }

        if let Some(sk) = self.skeleton.as_mut() {
            sk.set_data(data_arc.clone());
            sk.set_skin(Some(&name));
            sk.setup_pose();
        }
        self.ui.skin = name.clone();
        Ok(format!("Saved outfit as skin '{name}'"))
    }

    /// Re-registers unsaved custom skins after the skeleton data was reloaded from disk.
    fn reapply_custom_skins(&mut self, requested_skin: &str) {
        if self.custom_skins.is_empty() {
            return;
        }
        let Some(data_arc) = self.data.as_mut() else {
            return;
        };
        let data = Arc::make_mut(data_arc);
        for rec in &self.custom_skins {
            data.add_skin(rec.skin.clone());
        }
        if let Some(sk) = self.skeleton.as_mut() {
            sk.set_data(data_arc.clone());
            if self.custom_skins.iter().any(|r| r.name == requested_skin) {
                sk.set_skin(Some(requested_skin));
                self.ui.skin = requested_skin.to_string();
            }
        }
    }

    /// When a custom skin is selected, restore the outfit (overrides + PNG parts) it was built from.
    fn activate_custom_skin_outfit(&mut self, skin_name: &str) {
        let Some(rec) = self.custom_skins.iter().find(|r| r.name == skin_name).cloned() else {
            return;
        };
        self.slot_overrides = rec.overrides;
        for (slot, cfg) in rec.sprites {
            if !self.textures.bind_groups.contains_key(&format!("__custom_slot_{slot}__")) {
                let _ = self.upload_custom_texture(&slot, Path::new(&cfg.source_path));
            }
            self.custom_sprites.insert(slot, cfg);
        }
    }

    /// Uploads `path` as the GPU texture backing the custom part on `slot_name`.
    fn upload_custom_texture(
        &mut self,
        slot_name: &str,
        path: &Path,
    ) -> Result<(u32, u32), String> {
        let (w, h, pixels) = load_png_rgba8(path)?;
        let (Some(device), Some(queue), Some(renderer)) = (
            self.device.as_ref(),
            self.queue.as_ref(),
            self.renderer.as_ref(),
        ) else {
            return Err("Renderer not ready".to_string());
        };
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("custom sprite sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("custom sprite texture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            texture.as_image_copy(),
            &pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * w),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group =
            create_texture_bind_group(device, renderer.texture_bind_group_layout(), &view, &sampler);
        self.textures
            .bind_groups
            .insert(format!("__custom_slot_{slot_name}__"), bind_group);
        Ok((w, h))
    }

    fn create_new_slot(&mut self, slot_name: &str, bone_name: &str) -> Result<(), String> {
        let slot_name = slot_name.trim();
        let bone_name = bone_name.trim();
        if slot_name.is_empty() {
            return Err("Slot name cannot be empty".to_string());
        }
        let Some(data_arc) = self.data.as_mut() else {
            return Err("Skeleton data not loaded".to_string());
        };
        if data_arc.find_slot(slot_name).is_some() {
            self.custom_sprite_slot = slot_name.to_string();
            return Ok(());
        }
        let Some(bone_data) = data_arc.find_bone(bone_name) else {
            return Err(format!("Bone '{bone_name}' not found"));
        };
        let bone_idx = bone_data.get_index();

        let data = Arc::make_mut(data_arc);
        let new_slot_idx = data.add_slot(slot_name.to_string(), bone_idx, None);

        if let Some(sk) = self.skeleton.as_mut() {
            sk.set_data(data_arc.clone());
        }
        if let Some(order) = self.custom_draw_order.as_mut()
            && !order.contains(&new_slot_idx)
        {
            order.push(new_slot_idx);
        }
        if !self
            .created_slots
            .iter()
            .any(|(s, _)| s == slot_name)
        {
            self.created_slots
                .push((slot_name.to_string(), bone_name.to_string()));
        }
        self.custom_sprite_slot = slot_name.to_string();
        Ok(())
    }

    fn export_modified_spine(&mut self) -> Result<String, String> {
        let export_dir = self
            .atlas_path
            .parent()
            .ok_or_else(|| "No parent dir for atlas".to_string())?
            .to_path_buf();

        let json_str = std::fs::read_to_string(&self.json_path)
            .map_err(|e| format!("read json: {e}"))?;
        let mut root: serde_json::Value =
            serde_json::from_str(&json_str).map_err(|e| format!("parse json: {e}"))?;

        let mut atlas_str = std::fs::read_to_string(&self.atlas_path)
            .map_err(|e| format!("read atlas: {e}"))?;

        // 1. Apply created_bones, bone_parent_overrides, and bone_offsets to bones array in JSON
        if let Some(bones) = root.get_mut("bones").and_then(|v| v.as_array_mut()) {
            for (new_b_name, new_b_parent) in &self.created_bones {
                let exists = bones.iter().any(|b| {
                    b.get("name").and_then(|v| v.as_str()) == Some(new_b_name.as_str())
                });
                if !exists {
                    bones.push(serde_json::json!({
                        "name": new_b_name,
                        "parent": new_b_parent,
                        "x": 0.0,
                        "y": 0.0,
                        "rotation": 0.0,
                        "scaleX": 1.0,
                        "scaleY": 1.0,
                        "length": 50.0
                    }));
                }
            }

            for bone in bones {
                let Some(name) = bone.get("name").and_then(|v| v.as_str()).map(str::to_string) else {
                    continue;
                };
                if let Some(new_p) = self.bone_parent_overrides.get(&name) {
                    if let Some(obj) = bone.as_object_mut() {
                        obj.insert("parent".to_string(), serde_json::json!(new_p));
                    }
                }
                if let Some(off) = self.bone_offsets.get(&name) {
                    let cur_x = bone.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                    let cur_y = bone.get("y").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                    let cur_rot =
                        bone.get("rotation").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                    let cur_sx = bone.get("scaleX").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                    let cur_sy = bone.get("scaleY").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;

                    if let Some(obj) = bone.as_object_mut() {
                        obj.insert("x".to_string(), serde_json::json!(cur_x + off.dx));
                        obj.insert("y".to_string(), serde_json::json!(cur_y + off.dy));
                        obj.insert(
                            "rotation".to_string(),
                            serde_json::json!(cur_rot + off.drot),
                        );
                        obj.insert("scaleX".to_string(), serde_json::json!(cur_sx * off.sx));
                        obj.insert("scaleY".to_string(), serde_json::json!(cur_sy * off.sy));
                        if off.length > 0.0 {
                            obj.insert("length".to_string(), serde_json::json!(off.length));
                        }
                    }
                }
            }
        }

        // 2. Apply created_slots, slot_bone_overrides, slot_overrides, and custom_draw_order to slots array in JSON
        if let Some(slots) = root.get_mut("slots").and_then(|v| v.as_array_mut()) {
            for (new_slot_name, bone_name) in &self.created_slots {
                let exists = slots.iter().any(|s| {
                    s.get("name").and_then(|v| v.as_str()) == Some(new_slot_name.as_str())
                });
                if !exists {
                    slots.push(serde_json::json!({
                        "name": new_slot_name,
                        "bone": bone_name
                    }));
                }
            }

            let bone_names: Vec<String> = self
                .data
                .as_ref()
                .map(|d| {
                    d.get_bones()
                        .iter()
                        .map(|b| b.get_name().to_string())
                        .collect()
                })
                .unwrap_or_default();

            for slot in slots.iter_mut() {
                let Some(slot_name) =
                    slot.get("name").and_then(|v| v.as_str()).map(str::to_string)
                else {
                    continue;
                };
                if let Some(&b_idx) = self.slot_bone_overrides.get(&slot_name)
                    && let Some(b_name) = bone_names.get(b_idx)
                    && let Some(obj) = slot.as_object_mut()
                {
                    obj.insert("bone".to_string(), serde_json::json!(b_name));
                }
                if let Some(ov) = self.slot_overrides.get(&slot_name)
                    && let Some(obj) = slot.as_object_mut()
                {
                    match ov {
                        SlotOverride::Hidden => {
                            obj.remove("attachment");
                        }
                        SlotOverride::Attachment { attachment, .. } => {
                            obj.insert("attachment".to_string(), serde_json::json!(attachment));
                        }
                    }
                }
            }

            if let Some(order) = &self.custom_draw_order
                && order.len() == slots.len()
            {
                let original = slots.clone();
                for (new_pos, &old_idx) in order.iter().enumerate() {
                    if let Some(item) = original.get(old_idx).cloned() {
                        slots[new_pos] = item;
                    }
                }
            }
        }

        let active_skin_name = self
            .skeleton
            .as_ref()
            .and_then(|sk| sk.get_skin().map(|s| s.get_name().to_string()))
            .unwrap_or_else(|| "default".to_string());

        // 2b. Bake custom skins built with the Skin Builder into the JSON `skins` array.
        if let Some(skins) = root.get_mut("skins").and_then(|v| v.as_array_mut()) {
            for rec in &self.custom_skins {
                let base = skins
                    .iter()
                    .find(|s| s.get("name").and_then(|v| v.as_str()) == Some(rec.base_skin.as_str()))
                    .cloned();
                let mut new_skin = base.unwrap_or_else(|| serde_json::json!({ "attachments": {} }));
                if let Some(obj) = new_skin.as_object_mut() {
                    obj.insert("name".to_string(), serde_json::json!(rec.name));
                    obj.entry("attachments".to_string())
                        .or_insert_with(|| serde_json::json!({}));
                }
                skins.retain(|s| s.get("name").and_then(|v| v.as_str()) != Some(rec.name.as_str()));
                skins.push(new_skin);
            }
        }

        // 3. Inject custom PNGs into atlas + JSON (default skin, active skin and custom skins).
        for (slot_name, cfg) in &self.custom_sprites {
            let (region_name, w, h) =
                ensure_atlas_region(&mut atlas_str, &export_dir, slot_name, cfg);
            if let Some(skins) = root.get_mut("skins").and_then(|v| v.as_array_mut()) {
                for skin in skins {
                    let s_name = skin.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    if s_name == "default" || s_name == active_skin_name {
                        json_skin_set_custom_region(skin, slot_name, &region_name, cfg, w, h);
                    }
                }
            }
        }
        for rec in &self.custom_skins {
            for (slot_name, cfg) in &rec.sprites {
                let (region_name, w, h) =
                    ensure_atlas_region(&mut atlas_str, &export_dir, slot_name, cfg);
                if let Some(skins) = root.get_mut("skins").and_then(|v| v.as_array_mut()) {
                    for skin in skins {
                        if skin.get("name").and_then(|v| v.as_str()) == Some(rec.name.as_str()) {
                            json_skin_set_custom_region(skin, slot_name, &region_name, cfg, w, h);
                        }
                    }
                }
            }
        }

        // Bake modified existing part transforms into the active skin and default skin in JSON
        for (slot_name, trans) in &self.part_transforms {
            if let Some(skins) = root.get_mut("skins").and_then(|v| v.as_array_mut()) {
                for skin in skins {
                    let s_name = skin.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    if s_name == "default" || s_name == active_skin_name {
                        if let Some(attachments) = skin.get_mut("attachments").and_then(|v| v.as_object_mut()) {
                            if let Some(slot_att) = attachments.get_mut(slot_name).and_then(|v| v.as_object_mut()) {
                                for (_att_name, att_val) in slot_att.iter_mut() {
                                    if let Some(att_obj) = att_val.as_object_mut() {
                                        att_obj.insert("x".to_string(), serde_json::json!(trans.x));
                                        att_obj.insert("y".to_string(), serde_json::json!(trans.y));
                                        att_obj.insert("rotation".to_string(), serde_json::json!(trans.rotation));
                                        att_obj.insert("scaleX".to_string(), serde_json::json!(trans.scale_x));
                                        att_obj.insert("scaleY".to_string(), serde_json::json!(trans.scale_y));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Copy cross-skin overridden attachments into the active skin / custom skins in JSON so the
        // exported JSON works standalone.
        if let Some(skins) = root.get_mut("skins").and_then(|v| v.as_array_mut()) {
            copy_cross_skin_overrides(skins, &self.slot_overrides, &active_skin_name);
            for rec in &self.custom_skins {
                copy_cross_skin_overrides(skins, &rec.overrides, &rec.name);
            }
        }

        std::fs::write(&self.atlas_path, &atlas_str)
            .map_err(|e| format!("write atlas {:?}: {e}", self.atlas_path))?;

        // Also write .atlas.txt for direct Spine-Unity importer compatibility (Unity requires .atlas.txt)
        let atlas_txt_path = if self.atlas_path.to_string_lossy().ends_with(".atlas.txt") {
            self.atlas_path.clone()
        } else {
            let stem = self.atlas_path.file_stem().unwrap_or_default().to_string_lossy();
            self.atlas_path.with_file_name(format!("{}.atlas.txt", stem))
        };
        let _ = std::fs::write(&atlas_txt_path, &atlas_str);

        let out_json =
            serde_json::to_string_pretty(&root).map_err(|e| format!("serialize json: {e}"))?;
        std::fs::write(&self.json_path, out_json)
            .map_err(|e| format!("write json {:?}: {e}", self.json_path))?;

        Ok(format!(
            "Saved & recompiled export to {:?} (and .atlas.txt for Unity)",
            self.json_path.file_name().unwrap_or_default()
        ))
    }

    fn apply_part_transform(&mut self, slot_name: &str, trans: &PartSpriteTransform) {
        let Some(data_arc) = self.data.as_mut() else { return; };
        let Some(slot_data) = data_arc.find_slot(slot_name) else { return; };
        let slot_idx = slot_data.get_index();
        let active_skin_name = self.skeleton.as_ref()
            .and_then(|sk| sk.get_skin().map(|s| s.get_name().to_string()))
            .unwrap_or_else(|| "default".to_string());

        let data = Arc::make_mut(data_arc);
        for s_name in [&active_skin_name, "default"] {
            if let Some(skin) = data.find_skin_mut(s_name) {
                let mut att_names = Vec::new();
                skin.find_names_for_slot(slot_idx, &mut att_names);
                for a_name in att_names {
                    if let Some(att) = skin.get_attachment_mut(slot_idx, &a_name) {
                        if let spine2d::AttachmentData::Region(r) = att {
                            r.set_x(trans.x);
                            r.set_y(trans.y);
                            r.set_rotation(trans.rotation);
                            r.set_scale_x(trans.scale_x);
                            r.set_scale_y(trans.scale_y);
                            r.get_color_mut()[3] = trans.alpha;
                        }
                    }
                }
            }
        }
        if let Some(sk) = self.skeleton.as_mut() {
            sk.set_data(data_arc.clone());
        }
    }

    fn update_custom_sprite_attachment(&mut self, slot_name: &str, cfg: &CustomSpriteConfig) {
        let Some(data_arc) = self.data.as_mut() else {
            return;
        };
        let Some(slot_data) = data_arc.find_slot(slot_name) else {
            return;
        };
        let slot_idx = slot_data.get_index();
        let data = Arc::make_mut(data_arc);
        let Some(default_skin) = data
            .get_skins_mut()
            .iter_mut()
            .find(|s| s.get_name() == "default")
        else {
            return;
        };

        let tex_key = format!("__custom_slot_{slot_name}__");
        let mut reg =
            spine2d::RegionAttachmentData::new(tex_key.clone(), tex_key.clone(), cfg.width, cfg.height);
        reg.set_x(cfg.x);
        reg.set_y(cfg.y);
        reg.set_rotation(cfg.rotation);
        reg.set_scale_x(cfg.scale_x);
        reg.set_scale_y(cfg.scale_y);
        default_skin.set_attachment(slot_idx, tex_key, spine2d::AttachmentData::Region(reg));

        if let Some(sk) = self.skeleton.as_mut() {
            sk.set_data(data_arc.clone());
        }
    }

    fn apply_pending_custom_png(&mut self) {
        let Some((slot_name, raw_path)) = self.pending_custom_png.take() else {
            return;
        };
        let tex_key = format!("__custom_slot_{slot_name}__");
        let path = PathBuf::from(raw_path.trim().trim_matches('"'));
        let (w, h) = match self.upload_custom_texture(&slot_name, &path) {
            Ok(v) => v,
            Err(e) => {
                self.last_error = Some(e);
                return;
            }
        };
        let Some(data_arc) = self.data.as_ref() else {
            return;
        };

        // Auto-adapt position, rotation, and scale from the active skin's existing attachment on that slot
        let mut default_x = 0.0;
        let mut default_y = 0.0;
        let mut default_rot = 0.0;
        let mut default_sx = 1.0;
        let mut default_sy = 1.0;
        let active_skin_name = self
            .skeleton
            .as_ref()
            .and_then(|sk| sk.get_skin().map(|s| s.get_name().to_string()))
            .unwrap_or_default();

        if let Some(slot_data) = data_arc.find_slot(&slot_name) {
            let slot_idx = slot_data.get_index();
            // Prefer the active skin's attachment first, then any skin
            let mut ref_region: Option<&spine2d::RegionAttachmentData> = None;
            if let Some(active_skin) = data_arc.find_skin(&active_skin_name) {
                let mut att_list = Vec::new();
                active_skin.find_attachments_for_slot(slot_idx, &mut att_list);
                for att in att_list {
                    if !att.get_name().starts_with("__custom_slot_")
                        && let spine2d::AttachmentData::Region(r) = att
                    {
                        ref_region = Some(r);
                        break;
                    }
                }
            }
            if ref_region.is_none() {
                for skin in data_arc.get_skins() {
                    let mut att_list = Vec::new();
                    skin.find_attachments_for_slot(slot_idx, &mut att_list);
                    for att in att_list {
                        if !att.get_name().starts_with("__custom_slot_")
                            && let spine2d::AttachmentData::Region(r) = att
                        {
                            ref_region = Some(r);
                            break;
                        }
                    }
                    if ref_region.is_some() {
                        break;
                    }
                }
            }

            if let Some(r) = ref_region {
                default_x = r.get_x();
                default_y = r.get_y();
                default_rot = r.get_rotation();
                let ref_w = (r.get_width() * r.get_scale_x().abs()).max(1.0);
                let ref_h = (r.get_height() * r.get_scale_y().abs()).max(1.0);
                default_sx = (ref_w / (w.max(1) as f32)) * r.get_scale_x().signum();
                default_sy = (ref_h / (h.max(1) as f32)) * r.get_scale_y().signum();
            }
        }

        let cfg = CustomSpriteConfig {
            source_path: path.display().to_string(),
            x: default_x,
            y: default_y,
            rotation: default_rot,
            scale_x: default_sx,
            scale_y: default_sy,
            width: w as f32,
            height: h as f32,
        };
        self.selected_part = slot_name.clone();
        self.update_custom_sprite_attachment(&slot_name, &cfg);
        self.custom_sprites.insert(slot_name.clone(), cfg);
        self.slot_overrides.insert(
            slot_name,
            SlotOverride::Attachment {
                skin: "default".to_string(),
                attachment: tex_key,
            },
        );
        self.last_error = None;
    }

    fn apply_runtime_edits(
        skeleton: &mut Skeleton,
        data: &SkeletonData,
        slot_overrides: &HashMap<String, SlotOverride>,
        slot_bone_overrides: &HashMap<String, usize>,
        bone_offsets: &HashMap<String, BoneOffset>,
        custom_draw_order: Option<&Vec<usize>>,
    ) {
        for (slot_name, ov) in slot_overrides {
            match ov {
                SlotOverride::Hidden => {
                    skeleton.set_attachment(slot_name, "");
                }
                SlotOverride::Attachment { skin, attachment } => {
                    skeleton.set_attachment_from_skin(slot_name, attachment, skin);
                }
            }
        }

        if !slot_bone_overrides.is_empty() {
            let slot_names: Vec<String> = data
                .get_slots()
                .iter()
                .map(|s| s.get_name().to_string())
                .collect();
            let slots = skeleton.get_slots_mut();
            for (idx, name) in slot_names.iter().enumerate() {
                if let Some(&bone_idx) = slot_bone_overrides.get(name)
                    && let Some(slot) = slots.get_mut(idx)
                {
                    slot.set_bone_index(bone_idx);
                }
            }
        }

        if !bone_offsets.is_empty() {
            let bone_names: Vec<String> = data
                .get_bones()
                .iter()
                .map(|b| b.get_name().to_string())
                .collect();
            let bones = skeleton.get_bones_mut();
            for (idx, name) in bone_names.iter().enumerate() {
                if let Some(off) = bone_offsets.get(name)
                    && let Some(bone) = bones.get_mut(idx)
                {
                    bone.set_x(bone.get_x() + off.dx);
                    bone.set_y(bone.get_y() + off.dy);
                    bone.set_rotation(bone.get_rotation() + off.drot);
                    bone.set_scale_x(bone.get_scale_x() * off.sx);
                    bone.set_scale_y(bone.get_scale_y() * off.sy);
                }
            }
        }

        if let Some(order) = custom_draw_order {
            let dst = skeleton.get_draw_order_mut();
            if dst.len() == order.len() {
                dst.copy_from_slice(order);
            }
        }
    }

    fn recompute_fit(&mut self) {
        let (Some(renderer), Some(queue), Some(config)) = (
            self.renderer.as_ref(),
            self.queue.as_ref(),
            self.config.as_ref(),
        ) else {
            return;
        };

        if let Some(bounds) = self.initial_bounds {
            renderer.update_globals_matrix(
                queue,
                clip_from_world_fit_bounds(bounds, config.width, config.height, self.fit_margin),
            );
        } else {
            renderer.update_globals_ortho_centered(
                queue,
                config.width as f32,
                config.height as f32,
            );
        }
    }

    fn queue_reload(&mut self) {
        self.pending_reload = Some(self.ui.clone());
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }

    fn ui_panel(&mut self, ctx: &egui::Context) {
        egui::Window::new("spine2d Studio")
            .default_pos([8.0, 8.0])
            .default_size([480.0, 680.0])
            .min_width(320.0)
            .min_height(240.0)
            .resizable(true)
            .vscroll(true)
            .hscroll(true)
            .collapsible(true)
            .show(ctx, |ui| {
                ui.label("Controls");
                ui.separator();


                egui::Grid::new("spine2d_controls_grid")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Example");
                        let mut example = self.ui.example.clone();
                        ui.horizontal(|ui| {
                            egui::ComboBox::from_id_salt("example_combo")
                                .width(150.0)
                                .selected_text(&example)
                                .show_ui(ui, |ui| {
                                    for name in &self.examples {
                                        ui.selectable_value(&mut example, name.clone(), name);
                                    }
                                });
                            if ui
                                .button("📥 Import...")
                                .on_hover_text("Import any external Spine export file (.json, .skel, .skel.bytes, .atlas). Safely copies to examples/ without touching your original files!")
                                .clicked()
                            {
                                self.pick_and_import_file();
                            }
                            if ui
                                .button("📁")
                                .on_hover_text("Import external character folder (safely copies to examples/)")
                                .clicked()
                            {
                                self.pick_and_import_folder();
                            }
                        });
                        if example != self.ui.example {
                            self.ui.example = example;
                            // New skeleton almost certainly has a different animation/skin set.
                            // Reset to "(auto)" so we don't carry an invalid name across examples.
                            self.clear_character_edits();
                            self.queue_reload();
                        }
                        ui.end_row();

                        ui.label("Anim");
                        let anims = self
                            .data
                            .as_ref()
                            .map(|d| {
                                d.get_animations()
                                    .iter()
                                    .map(|a| a.get_name().to_string())
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default();
                        let mut anims = anims;
                        anims.sort();
                        let mut animation = self.ui.animation.clone();
                        egui::ComboBox::from_id_salt("anim_combo")
                            .width(260.0)
                            .selected_text(if animation.is_empty() {
                                "(auto)"
                            } else {
                                &animation
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut animation, String::new(), "(auto)");
                                for name in &anims {
                                    ui.selectable_value(&mut animation, name.clone(), name);
                                }
                            });
                        if animation != self.ui.animation {
                            self.ui.animation = animation;
                            let chosen = if !self.ui.animation.is_empty() {
                                Some(self.ui.animation.clone())
                            } else {
                                self.data.as_ref().and_then(|d| choose_default_animation(d))
                            };
                            if let (Some(anim_name), Some(state)) = (chosen, self.state.as_mut()) {
                                state.set_animation(0, anim_name.as_str(), true);
                                self.animation = Some(anim_name);
                            }
                        }
                        ui.end_row();

                        ui.label("Skin");
                        let skins = self
                            .data
                            .as_ref()
                            .map(|d| {
                                d.get_skins()
                                    .iter()
                                    .map(|skin| skin.get_name().to_string())
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default();
                        let mut skins = skins;
                        skins.sort();
                        let mut skin = self.ui.skin.clone();
                        egui::ComboBox::from_id_salt("skin_combo")
                            .width(260.0)
                            .selected_text(match skin.as_str() {
                                "" => "(auto)",
                                SKIN_NONE_SENTINEL => "(none)",
                                _ => &skin,
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut skin, String::new(), "(auto)");
                                ui.selectable_value(
                                    &mut skin,
                                    SKIN_NONE_SENTINEL.to_string(),
                                    "(none)",
                                );
                                for name in &skins {
                                    ui.selectable_value(&mut skin, name.clone(), name);
                                }
                            });
                        if skin != self.ui.skin {
                            self.ui.skin = skin;
                            if let (Some(data), Some(sk)) =
                                (self.data.as_ref(), self.skeleton.as_mut())
                            {
                                let chosen = if self.ui.skin == SKIN_NONE_SENTINEL {
                                    None
                                } else if !self.ui.skin.is_empty()
                                    && data.find_skin(&self.ui.skin).is_some()
                                {
                                    Some(self.ui.skin.clone())
                                } else {
                                    choose_default_skin(&self.example, data)
                                };
                                sk.set_skin(chosen.as_deref());
                                sk.setup_pose();
                            }
                            let picked = self.ui.skin.clone();
                            self.activate_custom_skin_outfit(&picked);
                        }
                        ui.end_row();

                        ui.label("Speed");
                        if ui
                            .add(egui::Slider::new(&mut self.ui.speed, 0.0..=2.0).step_by(0.05))
                            .changed()
                        {
                            self.speed = self.ui.speed;
                        }
                        ui.end_row();

                        ui.label("Margin");
                        if ui
                            .add(egui::Slider::new(&mut self.ui.margin, 0.1..=1.0).step_by(0.01))
                            .changed()
                        {
                            self.fit_margin = self.ui.margin;
                            self.needs_fit = true;
                        }
                        ui.end_row();
                    });

                ui.separator();
                ui.horizontal(|ui| {
                    if ui
                        .checkbox(&mut self.ui.prefer_pma_atlas, "Prefer PMA")
                        .changed()
                    {
                        self.queue_reload();
                    }
                    if ui.button("Reload").clicked() {
                        self.queue_reload();
                    }
                    if ui.button("Re-fit").clicked() {
                        self.needs_fit = true;
                    }
                    if ui
                        .button(if self.paused { "Play" } else { "Pause" })
                        .clicked()
                    {
                        self.paused = !self.paused;
                    }
                    ui.checkbox(&mut self.show_bone_gizmos, "Show Bones");
                    if (!self.slot_overrides.is_empty()
                        || !self.slot_bone_overrides.is_empty()
                        || !self.bone_offsets.is_empty()
                        || self.custom_draw_order.is_some())
                        && ui.button("Reset All Edits").clicked()
                    {
                        self.slot_overrides.clear();
        self.part_transforms.clear();
                        self.slot_bone_overrides.clear();
                        self.bone_offsets.clear();
                        self.custom_draw_order = None;
                        self.save_status = None;
                    }
                    if ui.button("Save & Recompile Spine Export").clicked() {
                        match self.export_modified_spine() {
                            Ok(msg) => {
                                self.save_status = Some(msg);
                                self.last_error = None;
                                // Clear transient diffs that were baked into the JSON, then reload fresh
                                self.slot_overrides.clear();
        self.part_transforms.clear();
                                self.slot_bone_overrides.clear();
                                self.bone_offsets.clear();
                                self.custom_draw_order = None;
                                self.custom_sprites.clear();
                                self.created_slots.clear();
                                self.custom_skins.clear();
                                self.queue_reload();
                            }
                            Err(e) => {
                                self.last_error = Some(e);
                                self.save_status = None;
                            }
                        }
                    }
                });
                if let Some(msg) = self.save_status.as_deref() {
                    ui.colored_label(egui::Color32::LIGHT_GREEN, msg);
                }

                ui.separator();
                ui.separator();
                ui.horizontal_wrapped(|ui| {
                    ui.selectable_value(&mut self.active_tab, StudioTab::CurrentSprites, "🖼️ Built-in Sprites & Parts");
                    ui.selectable_value(&mut self.active_tab, StudioTab::WardrobeCustom, "👗 Custom PNG Wardrobe");
                    ui.selectable_value(&mut self.active_tab, StudioTab::BoneStudio, "🦴 Bone Studio");
                    ui.selectable_value(&mut self.active_tab, StudioTab::SkinBuilder, "🎨 Skin Builder");
                    ui.selectable_value(&mut self.active_tab, StudioTab::LayerOrder, "📚 Layer Draw Order");
                    ui.selectable_value(&mut self.active_tab, StudioTab::ImportExport, "💾 Import / Export");
                });
                ui.separator();

                match self.active_tab {

                    StudioTab::CurrentSprites => {
                        ui.strong("🖼️ Modify Setted Bone Sprites & Parts");
                        ui.label("Select any current part sprite on the character to adjust its local transform, wire bone, or swap attachment:");

                        if let Some(data) = self.data.as_ref() {
                            let slots = data.get_slots();
                            let slot_names: Vec<String> = slots.iter().map(|s| s.get_name().to_string()).collect();
                            if (self.selected_current_slot.is_empty() || !slot_names.contains(&self.selected_current_slot))
                                && let Some(first) = slot_names.first()
                            {
                                self.selected_current_slot = first.clone();
                            }

                            ui.label("Quick Select Part:");
                            ui.horizontal_wrapped(|ui| {
                                for (icon, label, s_name) in [
                                    ("👗", "Body", "body"),
                                    ("🖐️", "Left Hand", "l_hand_C2"),
                                    ("🖐️", "Right Hand", "r_hand_C2"),
                                    ("👤", "Head", "head"),
                                    ("💇", "Hair L", "hair_2_l"),
                                    ("💇", "Hair R", "hair_2_r"),
                                    ("👗", "R Skirt", "r_skirt"),
                                    ("👗", "L Skirt", "l_skirt"),
                                    ("🦵", "Left Leg", "l_leg"),
                                    ("🦵", "Right Leg", "r_leg"),
                                ] {
                                    if slot_names.iter().any(|s| s == s_name) {
                                        let is_sel = self.selected_current_slot == s_name;
                                        if ui.selectable_label(is_sel, format!("{icon} {label}")).clicked() {
                                            self.selected_current_slot = s_name.to_string();
                                        }
                                    }
                                }
                            });

                            ui.horizontal(|ui| {
                                ui.label("Filter Slot:");
                                ui.text_edit_singleline(&mut self.current_slot_filter);
                                if !self.current_slot_filter.is_empty() && ui.button("X").clicked() {
                                    self.current_slot_filter.clear();
                                }
                            });

                            let cur_filter = self.current_slot_filter.trim().to_ascii_lowercase();
                            ui.horizontal(|ui| {
                                ui.label("Part Slot:");
                                egui::ComboBox::from_id_salt("current_sprite_slot_combo")
                                    .width(240.0)
                                    .selected_text(&self.selected_current_slot)
                                    .show_ui(ui, |ui| {
                                        for s in slots {
                                            let sn = s.get_name();
                                            if !cur_filter.is_empty() && !sn.to_ascii_lowercase().contains(&cur_filter) {
                                                continue;
                                            }
                                            let b_name = s.get_bone(data).get_name();
                                            let display = format!("{sn}  (Bone: {b_name})");
                                            ui.selectable_value(&mut self.selected_current_slot, sn.to_string(), display);
                                        }
                                    });
                            });

                            let target_slot = self.selected_current_slot.clone();
                            if let Some(slot_data) = data.find_slot(&target_slot) {
                                let slot_idx = slot_data.get_index();
                                let default_bone_idx = slot_data.get_bone(data).get_index();
                                let bone_names: Vec<String> = data.get_bones().iter().map(|b| b.get_name().to_string()).collect();

                                let cur_b_idx = self.slot_bone_overrides.get(&target_slot).copied().unwrap_or(default_bone_idx);
                                let mut next_b_idx = cur_b_idx;

                                ui.separator();
                                ui.horizontal(|ui| {
                                    ui.label("Wired Bone:");
                                    egui::ComboBox::from_id_salt("current_part_wired_bone_combo")
                                        .width(160.0)
                                        .selected_text(bone_names.get(cur_b_idx).map(|s| s.as_str()).unwrap_or("?"))
                                        .show_ui(ui, |ui| {
                                            for (idx, b) in bone_names.iter().enumerate() {
                                                ui.selectable_value(&mut next_b_idx, idx, b);
                                            }
                                        });
                                    if next_b_idx != cur_b_idx {
                                        if next_b_idx == default_bone_idx {
                                            self.slot_bone_overrides.remove(&target_slot);
                                        } else {
                                            self.slot_bone_overrides.insert(target_slot.clone(), next_b_idx);
                                        }
                                    }
                                });

                                let is_hidden = matches!(self.slot_overrides.get(&target_slot), Some(SlotOverride::Hidden));
                                ui.horizontal(|ui| {
                                    if ui.button(if is_hidden { "👁️ Show Sprite" } else { "🚫 Hide Sprite" }).clicked() {
                                        if is_hidden {
                                            self.slot_overrides.remove(&target_slot);
                                        } else {
                                            self.slot_overrides.insert(target_slot.clone(), SlotOverride::Hidden);
                                        }
                                    }

                                    let mut candidates: Vec<(String, String)> = Vec::new();
                                    for skin in data.get_skins() {
                                        let skin_name = skin.get_name();
                                        let mut names = Vec::new();
                                        skin.find_names_for_slot(slot_idx, &mut names);
                                        names.sort();
                                        for att_name in names {
                                            if !candidates.iter().any(|(s, a)| s == skin_name && a == &att_name) {
                                                candidates.push((skin_name.to_string(), att_name));
                                            }
                                        }
                                    }

                                    if !candidates.is_empty() {
                                        let active_label = match self.slot_overrides.get(&target_slot) {
                                            Some(SlotOverride::Hidden) => "(Hidden)".to_string(),
                                            Some(SlotOverride::Attachment { skin, attachment }) => format!("{attachment} ({skin})"),
                                            None => slot_data.get_attachment_name().to_string(),
                                        };
                                        ui.label("Swap Attachment:");
                                        egui::ComboBox::from_id_salt("current_sprite_candidate_combo")
                                            .width(180.0)
                                            .selected_text(&active_label)
                                            .show_ui(ui, |ui| {
                                                if ui.selectable_label(!is_hidden && !self.slot_overrides.contains_key(&target_slot), "(default)").clicked() {
                                                    self.slot_overrides.remove(&target_slot);
                                                }
                                                for (sk_name, att_name) in &candidates {
                                                    let label = format!("{att_name} ({sk_name})");
                                                    let is_cur = match self.slot_overrides.get(&target_slot) {
                                                        Some(SlotOverride::Attachment { skin, attachment }) => skin == sk_name && attachment == att_name,
                                                        _ => false,
                                                    };
                                                    if ui.selectable_label(is_cur, label).clicked() {
                                                        self.slot_overrides.insert(
                                                            target_slot.clone(),
                                                            SlotOverride::Attachment {
                                                                skin: sk_name.clone(),
                                                                attachment: att_name.clone(),
                                                            },
                                                        );
                                                    }
                                                }
                                            });
                                    }
                                });

                                let active_skin_name = self.skeleton.as_ref()
                                    .and_then(|sk| sk.get_skin().map(|s| s.get_name().to_string()))
                                    .unwrap_or_else(|| "default".to_string());

                                let mut trans = self.part_transforms.get(&target_slot).copied().unwrap_or_else(|| {
                                    let mut t = PartSpriteTransform::default();
                                    for s_name in [&active_skin_name, "default"] {
                                        if let Some(skin) = data.find_skin(s_name) {
                                            let mut att_list = Vec::new();
                                            skin.find_attachments_for_slot(slot_idx, &mut att_list);
                                            for att in att_list {
                                                if let spine2d::AttachmentData::Region(r) = att {
                                                    t.x = r.get_x();
                                                    t.y = r.get_y();
                                                    t.rotation = r.get_rotation();
                                                    t.scale_x = r.get_scale_x();
                                                    t.scale_y = r.get_scale_y();
                                                    t.alpha = r.get_color()[3];
                                                    return t;
                                                }
                                            }
                                        }
                                    }
                                    t
                                });

                                let mut trans_changed = false;
                                ui.separator();
                                ui.strong(format!("Sprite Local Transform ('{target_slot}'):"));
                                egui::Grid::new("current_part_sprite_grid")
                                    .num_columns(2)
                                    .spacing([8.0, 4.0])
                                    .show(ui, |ui| {
                                        ui.label("Local X");
                                        ui.horizontal(|ui| {
                                            trans_changed |= ui.add(egui::Slider::new(&mut trans.x, -500.0..=500.0).step_by(0.5)).changed();
                                            if ui.button("-5").clicked() { trans.x -= 5.0; trans_changed = true; }
                                            if ui.button("+5").clicked() { trans.x += 5.0; trans_changed = true; }
                                            if ui.button("0").clicked() { trans.x = 0.0; trans_changed = true; }
                                        });
                                        ui.end_row();

                                        ui.label("Local Y");
                                        ui.horizontal(|ui| {
                                            trans_changed |= ui.add(egui::Slider::new(&mut trans.y, -500.0..=500.0).step_by(0.5)).changed();
                                            if ui.button("-5").clicked() { trans.y -= 5.0; trans_changed = true; }
                                            if ui.button("+5").clicked() { trans.y += 5.0; trans_changed = true; }
                                            if ui.button("0").clicked() { trans.y = 0.0; trans_changed = true; }
                                        });
                                        ui.end_row();

                                        ui.label("Rotation (deg)");
                                        ui.horizontal(|ui| {
                                            trans_changed |= ui.add(egui::Slider::new(&mut trans.rotation, -180.0..=180.0).step_by(0.5)).changed();
                                            if ui.button("-45°").clicked() { trans.rotation -= 45.0; trans_changed = true; }
                                            if ui.button("+45°").clicked() { trans.rotation += 45.0; trans_changed = true; }
                                            if ui.button("0°").clicked() { trans.rotation = 0.0; trans_changed = true; }
                                        });
                                        ui.end_row();

                                        ui.label("Scale X");
                                        ui.horizontal(|ui| {
                                            trans_changed |= ui.add(egui::Slider::new(&mut trans.scale_x, -3.0..=3.0).step_by(0.01)).changed();
                                            if ui.button("Flip X").clicked() { trans.scale_x = -trans.scale_x; trans_changed = true; }
                                            if ui.button("1.0").clicked() { trans.scale_x = 1.0; trans_changed = true; }
                                        });
                                        ui.end_row();

                                        ui.label("Scale Y");
                                        ui.horizontal(|ui| {
                                            trans_changed |= ui.add(egui::Slider::new(&mut trans.scale_y, -3.0..=3.0).step_by(0.01)).changed();
                                            if ui.button("Flip Y").clicked() { trans.scale_y = -trans.scale_y; trans_changed = true; }
                                            if ui.button("1.0").clicked() { trans.scale_y = 1.0; trans_changed = true; }
                                        });
                                        ui.end_row();

                                        ui.label("Opacity (Alpha)");
                                        ui.horizontal(|ui| {
                                            trans_changed |= ui.add(egui::Slider::new(&mut trans.alpha, 0.0..=1.0).step_by(0.05)).changed();
                                            if ui.button("100%").clicked() { trans.alpha = 1.0; trans_changed = true; }
                                            if ui.button("50%").clicked() { trans.alpha = 0.5; trans_changed = true; }
                                            if ui.button("0%").clicked() { trans.alpha = 0.0; trans_changed = true; }
                                        });
                                        ui.end_row();
                                    });

                                let mut do_reset = false;
                                let mut do_export = false;
                                ui.horizontal(|ui| {
                                    if ui.button("Reset Part Transform").clicked() {
                                        do_reset = true;
                                    }
                                    if ui.button("💾 Save & Recompile Spine Export").clicked() {
                                        do_export = true;
                                    }
                                });

                                if do_reset {
                                    for s_name in [&active_skin_name, "default"] {
                                        if let Some(skin) = data.find_skin(s_name) {
                                            let mut att_list = Vec::new();
                                            skin.find_attachments_for_slot(slot_idx, &mut att_list);
                                            for att in att_list {
                                                if let spine2d::AttachmentData::Region(r) = att {
                                                    trans.x = r.get_x();
                                                    trans.y = r.get_y();
                                                    trans.rotation = r.get_rotation();
                                                    trans.scale_x = r.get_scale_x();
                                                    trans.scale_y = r.get_scale_y();
                                                    trans.alpha = 1.0;
                                                    trans_changed = true;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }

                                if trans_changed {
                                    self.part_transforms.insert(target_slot.clone(), trans);
                                    self.apply_part_transform(&target_slot, &trans);
                                }
                                if do_reset {
                                    self.part_transforms.remove(&target_slot);
                                }
                                if do_export {
                                    match self.export_modified_spine() {
                                        Ok(msg) => {
                                            self.save_status = Some(msg);
                                            self.last_error = None;
                                        }
                                        Err(e) => {
                                            self.last_error = Some(e);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    StudioTab::WardrobeCustom => {
                    ui.horizontal_wrapped(|ui| {
                    ui.strong("Quick Hat:");
                    if ui.button("Add Beach Hat + Flower").clicked() {
                    self.slot_overrides.insert(
                    "hat2".to_string(),
                    SlotOverride::Attachment {
                    skin: "Mira_BeachPicnic".to_string(),
                    attachment: "hat".to_string(),
                    },
                    );
                    self.slot_overrides.insert(
                    "hat_flower".to_string(),
                    SlotOverride::Attachment {
                    skin: "Mira_BeachPicnic".to_string(),
                    attachment: "hat_flower".to_string(),
                    },
                    );
                    self.slot_overrides
                    .insert("hat".to_string(), SlotOverride::Hidden);
                    }
                    if ui.button("Add HotSpring Towel Hat").clicked() {
                    self.slot_overrides.insert(
                    "hat".to_string(),
                    SlotOverride::Attachment {
                    skin: "Mira_HotSpring".to_string(),
                    attachment: "hat".to_string(),
                    },
                    );
                    self.slot_overrides
                    .insert("hat2".to_string(), SlotOverride::Hidden);
                    self.slot_overrides
                    .insert("hat_flower".to_string(), SlotOverride::Hidden);
                    }
                    if ui.button("Remove Hat").clicked() {
                    self.slot_overrides
                    .insert("hat".to_string(), SlotOverride::Hidden);
                    self.slot_overrides
                    .insert("hat2".to_string(), SlotOverride::Hidden);
                    self.slot_overrides
                    .insert("hat_flower".to_string(), SlotOverride::Hidden);
                    }
                    });

                    ui.separator();
                    egui::CollapsingHeader::new(
                    "Wardrobe & Custom Part Studio (Hat, Clothes, Accessories, New Slots)",
                    )
                    .default_open(true)
                    .show(ui, |ui| {
                    let slot_names: Vec<String> = self
                    .data
                    .as_ref()
                    .map(|d| {
                    d.get_slots()
                    .iter()
                    .map(|s| s.get_name().to_string())
                    .collect()
                    })
                    .unwrap_or_default();

                    if (self.custom_sprite_slot.is_empty()
                    || !slot_names.contains(&self.custom_sprite_slot))
                    && let Some(first) = slot_names
                    .iter()
                    .find(|s| s.as_str() == "hat")
                    .or_else(|| slot_names.first())
                    {
                    self.custom_sprite_slot = first.clone();
                    }

                    ui.label("Part preset:");
                    ui.horizontal_wrapped(|ui| {
                    for (label, bone, slot, front) in WARDROBE_PRESETS {
                    if ui.button(*label).clicked() {
                    self.new_slot_name = slot.to_string();
                    self.new_slot_bone = bone.to_string();
                    self.new_slot_layer_front = *front;
                    if slot_names.iter().any(|s| s.as_str() == *slot) {
                    self.custom_sprite_slot = slot.to_string();
                    } else if self.create_new_slot(slot, bone).is_ok() {
                    self.set_slot_layer(slot, *front);
                    }
                    }
                    }
                    });

                    ui.horizontal(|ui| {
                    ui.label("Create Slot:");
                    ui.add(
                    egui::TextEdit::singleline(&mut self.new_slot_name)
                    .desired_width(110.0)
                    .hint_text("custom_hat"),
                    );
                    ui.label("on Bone:");
                    let bone_names: Vec<String> = self
                    .data
                    .as_ref()
                    .map(|d| {
                    d.get_bones()
                    .iter()
                    .map(|b| b.get_name().to_string())
                    .collect()
                    })
                    .unwrap_or_default();
                    if (self.new_slot_bone.is_empty()
                    || !bone_names.contains(&self.new_slot_bone))
                    && let Some(first) = bone_names
                    .iter()
                    .find(|b| b.as_str() == "head")
                    .or_else(|| bone_names.first())
                    {
                    self.new_slot_bone = first.clone();
                    }
                    egui::ComboBox::from_id_salt("new_slot_bone_combo")
                    .width(110.0)
                    .selected_text(&self.new_slot_bone)
                    .show_ui(ui, |ui| {
                    for b in &bone_names {
                    ui.selectable_value(
                    &mut self.new_slot_bone,
                    b.clone(),
                    b,
                    );
                    }
                    });
                    if ui.button("+ Add New Slot").clicked() {
                    let s_name = self.new_slot_name.clone();
                    let b_name = self.new_slot_bone.clone();
                    let front = self.new_slot_layer_front;
                    match self.create_new_slot(&s_name, &b_name) {
                    Ok(()) => {
                    self.set_slot_layer(s_name.trim(), front);
                    self.save_status = Some(format!(
                    "Created slot '{}' on bone '{b_name}' ({}) - target selected",
                    s_name.trim(),
                    if front { "front" } else { "back" }
                    ));
                    self.last_error = None;
                    }
                    Err(e) => {
                    self.last_error = Some(e);
                    }
                    }
                    }
                    });
                    ui.horizontal(|ui| {
                    ui.label("Layer:");
                    ui.selectable_value(
                    &mut self.new_slot_layer_front,
                    true,
                    "Front (Top)",
                    );
                    ui.selectable_value(
                    &mut self.new_slot_layer_front,
                    false,
                    "Back (Behind Character)",
                    );
                    });

                    ui.horizontal(|ui| {
                    ui.label("Target Slot:");
                    egui::ComboBox::from_id_salt("custom_png_slot_combo")
                    .width(160.0)
                    .selected_text(&self.custom_sprite_slot)
                    .show_ui(ui, |ui| {
                    for name in &slot_names {
                    ui.selectable_value(
                    &mut self.custom_sprite_slot,
                    name.clone(),
                    name,
                    );
                    }
                    });
                    });

                    ui.horizontal(|ui| {
                    ui.label("PNG File:");
                    ui.add(
                    egui::TextEdit::singleline(&mut self.custom_sprite_path)
                    .desired_width(220.0)
                    .hint_text("C:\\path\\to\\hat.png"),
                    );
                    if ui.button("Browse PNG...").clicked() {
                    if let Ok(out) = std::process::Command::new("powershell")
                    .args([
                    "-NoProfile",
                    "-Command",
                    "Add-Type -AssemblyName System.Windows.Forms; $f = New-Object System.Windows.Forms.OpenFileDialog; $f.Filter = 'PNG Images (*.png)|*.png|All Files (*.*)|*.*'; if ($f.ShowDialog() -eq 'OK') { Write-Output $f.FileName }",
                    ])
                    .output()
                    {
                    let picked = String::from_utf8_lossy(&out.stdout).trim().to_string();
                    if !picked.is_empty() {
                    self.custom_sprite_path = picked.clone();
                    self.pending_custom_png = Some((
                    self.custom_sprite_slot.clone(),
                    picked,
                    ));
                    }
                    }
                    }
                    if ui.button("Load onto Slot").clicked()
                    && !self.custom_sprite_path.trim().is_empty()
                    {
                    self.pending_custom_png = Some((
                    self.custom_sprite_slot.clone(),
                    self.custom_sprite_path.clone(),
                    ));
                    }
                    });

                    ui.separator();
                    ui.strong(format!(
                    "Active Custom Parts ({} loaded)",
                    self.custom_sprites.len()
                    ));
                    let mut parts: Vec<String> = self.custom_sprites.keys().cloned().collect();
                    parts.sort();
                    if !parts.contains(&self.selected_part) {
                    self.selected_part = parts.first().cloned().unwrap_or_default();
                    }
                    if parts.is_empty() {
                    ui.weak("No custom PNG parts yet. Pick a preset, add a slot, then load a PNG.");
                    }
                    ui.horizontal_wrapped(|ui| {
                    for part in &parts {
                    if ui
                    .selectable_label(self.selected_part == *part, part)
                    .clicked()
                    {
                    self.selected_part = part.clone();
                    self.custom_sprite_slot = part.clone();
                    }
                    }
                    });

                    let target_slot = self.selected_part.clone();
                    if let Some(mut cfg) = self.custom_sprites.get(&target_slot).cloned() {
                    ui.label(format!("Part '{target_slot}': {}", cfg.source_path));
                    let mut changed = false;
                    egui::Grid::new("custom_sprite_transform_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                    ui.label("Local X");
                    changed |= ui
                    .add(
                    egui::Slider::new(&mut cfg.x, -600.0..=600.0)
                    .step_by(1.0),
                    )
                    .changed();
                    ui.end_row();

                    ui.label("Local Y");
                    changed |= ui
                    .add(
                    egui::Slider::new(&mut cfg.y, -600.0..=600.0)
                    .step_by(1.0),
                    )
                    .changed();
                    ui.end_row();

                    ui.label("Rotation");
                    changed |= ui
                    .add(
                    egui::Slider::new(&mut cfg.rotation, -180.0..=180.0)
                    .step_by(0.5),
                    )
                    .changed();
                    ui.end_row();

                    ui.label("Scale X");
                    changed |= ui
                    .add(
                    egui::Slider::new(&mut cfg.scale_x, -4.0..=4.0)
                    .step_by(0.02),
                    )
                    .changed();
                    ui.end_row();

                    ui.label("Scale Y");
                    changed |= ui
                    .add(
                    egui::Slider::new(&mut cfg.scale_y, -4.0..=4.0)
                    .step_by(0.02),
                    )
                    .changed();
                    ui.end_row();
                    });
                    let is_hand_slot = target_slot.contains("hand")
                    || target_slot.contains("glove")
                    || target_slot.contains("arm");
                    if is_hand_slot {
                    ui.separator();
                    ui.strong("🖐️ Hand Alignment & Double-Hand Fix:");
                    ui.horizontal_wrapped(|ui| {
                    let counterpart_slot = if target_slot.contains('l') || target_slot.starts_with('l') {
                    "l_hand_C2"
                    } else {
                    "r_hand_C2"
                    };
                    let is_hidden = matches!(
                    self.slot_overrides.get(counterpart_slot),
                    Some(SlotOverride::Hidden)
                    );
                    if ui
                    .button(if is_hidden {
                    format!("👁️ Show Underlying Hand ('{counterpart_slot}')")
                    } else {
                    format!("🚫 Hide Underlying Hand ('{counterpart_slot}') [Fix Double Hand]")
                    })
                    .clicked()
                    {
                    if is_hidden {
                    self.slot_overrides.remove(counterpart_slot);
                    } else {
                    self.slot_overrides.insert(
                    counterpart_slot.to_string(),
                    SlotOverride::Hidden,
                    );
                    }
                    }

                    if ui.button("📐 Auto-Align to Natural Hand Angle").clicked() {
                    if counterpart_slot == "l_hand_C2" {
                    cfg.rotation = 40.62;
                    } else {
                    cfg.rotation = 138.54;
                    }
                    changed = true;
                    }

                    if ui.button("↔️ Flip Glove (Horizontal)").clicked() {
                    cfg.scale_x = -cfg.scale_x;
                    changed = true;
                    }
                    });

                    ui.horizontal(|ui| {
                    ui.label("Hand Nudge:");
                    if ui.button("⬅️ -5 X").clicked() {
                    cfg.x -= 5.0;
                    changed = true;
                    }
                    if ui.button("➡️ +5 X").clicked() {
                    cfg.x += 5.0;
                    changed = true;
                    }
                    if ui.button("⬇️ -5 Y").clicked() {
                    cfg.y -= 5.0;
                    changed = true;
                    }
                    if ui.button("⬆️ +5 Y").clicked() {
                    cfg.y += 5.0;
                    changed = true;
                    }
                    });
                    }

                    if changed {
                    self.update_custom_sprite_attachment(&target_slot, &cfg);
                    self.custom_sprites.insert(target_slot.clone(), cfg.clone());
                    }

                    // Wired bone for the selected part's slot.
                    let wiring = self.data.as_ref().and_then(|d| {
                    let slot = d.find_slot(&target_slot)?;
                    let default_idx = slot.get_bone(d).get_index();
                    let names: Vec<String> = d
                    .get_bones()
                    .iter()
                    .map(|b| b.get_name().to_string())
                    .collect();
                    Some((default_idx, names))
                    });
                    if let Some((default_idx, names)) = wiring {
                    let cur_idx = self
                    .slot_bone_overrides
                    .get(&target_slot)
                    .copied()
                    .unwrap_or(default_idx);
                    let mut next_idx = cur_idx;
                    ui.horizontal(|ui| {
                    ui.label("Wired Bone:");
                    egui::ComboBox::from_id_salt("custom_part_wired_bone")
                    .width(160.0)
                    .selected_text(
                    names.get(cur_idx).map(|s| s.as_str()).unwrap_or("?"),
                    )
                    .show_ui(ui, |ui| {
                    for (b_idx, b_name) in names.iter().enumerate() {
                    ui.selectable_value(&mut next_idx, b_idx, b_name);
                    }
                    });
                    });
                    if next_idx != cur_idx {
                    if next_idx == default_idx {
                    self.slot_bone_overrides.remove(&target_slot);
                    } else {
                    self.slot_bone_overrides
                    .insert(target_slot.clone(), next_idx);
                    }
                    }

                    if let Some(wired_bone_name) = names.get(next_idx).cloned() {
                    let mut bone_off = self
                    .bone_offsets
                    .get(&wired_bone_name)
                    .copied()
                    .unwrap_or_default();
                    let mut bone_changed = false;
                    ui.separator();
                    ui.strong(format!("Modify Wired Bone ('{wired_bone_name}'):"));
                    egui::Grid::new("inline_wired_bone_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                    ui.label("Bone Offset X");
                    bone_changed |= ui
                    .add(
                    egui::Slider::new(
                    &mut bone_off.dx,
                    -400.0..=400.0,
                    )
                    .step_by(0.5),
                    )
                    .changed();
                    ui.end_row();

                    ui.label("Bone Offset Y");
                    bone_changed |= ui
                    .add(
                    egui::Slider::new(
                    &mut bone_off.dy,
                    -400.0..=400.0,
                    )
                    .step_by(0.5),
                    )
                    .changed();
                    ui.end_row();

                    ui.label("Bone Rotate");
                    bone_changed |= ui
                    .add(
                    egui::Slider::new(
                    &mut bone_off.drot,
                    -180.0..=180.0,
                    )
                    .step_by(0.5),
                    )
                    .changed();
                    ui.end_row();

                    ui.label("Bone Scale X");
                    bone_changed |= ui
                    .add(
                    egui::Slider::new(
                    &mut bone_off.sx,
                    -3.0..=3.0,
                    )
                    .step_by(0.01),
                    )
                    .changed();
                    ui.end_row();

                    ui.label("Bone Scale Y");
                    bone_changed |= ui
                    .add(
                    egui::Slider::new(
                    &mut bone_off.sy,
                    -3.0..=3.0,
                    )
                    .step_by(0.01),
                    )
                    .changed();
                    ui.end_row();
                    });
                    ui.horizontal(|ui| {
                    if ui.button("Reset Bone Transform").clicked() {
                    bone_off = BoneOffset::default();
                    bone_changed = true;
                    }
                    if let Some(other_hand_slot) = match target_slot.as_str() {
                    "l_hand_C2" | "custom_l_hand" => Some("l_hand_C2"),
                    "r_hand_C2" | "custom_r_hand" => Some("r_hand_C2"),
                    _ => None,
                    } && other_hand_slot != target_slot.as_str()
                    {
                    let hidden = matches!(
                    self.slot_overrides.get(other_hand_slot),
                    Some(SlotOverride::Hidden)
                    );
                    if ui
                    .button(if hidden {
                    format!("Show Original '{other_hand_slot}'")
                    } else {
                    format!("Hide Original '{other_hand_slot}'")
                    })
                    .clicked()
                    {
                    if hidden {
                    self.slot_overrides.remove(other_hand_slot);
                    } else {
                    self.slot_overrides.insert(
                    other_hand_slot.to_string(),
                    SlotOverride::Hidden,
                    );
                    }
                    }
                    }
                    });
                    if bone_changed {
                    if bone_off.is_identity() {
                    self.bone_offsets.remove(&wired_bone_name);
                    } else {
                    self.bone_offsets.insert(wired_bone_name, bone_off);
                    }
                    }
                    }
                    }

                    let mut remove_part = false;
                    ui.horizontal_wrapped(|ui| {
                    if ui.button("Auto-Adapt to Original Slot").clicked() {
                    if let Some(data_arc) = self.data.as_ref()
                    && let Some(slot_data) = data_arc.find_slot(&target_slot)
                    {
                    let slot_idx = slot_data.get_index();
                    let mut found = None;
                    for skin in data_arc.get_skins() {
                    let mut att_list = Vec::new();
                    skin.find_attachments_for_slot(slot_idx, &mut att_list);
                    for att in att_list {
                    if !att.get_name().starts_with("__custom_slot_")
                    && let spine2d::AttachmentData::Region(r) = att
                    {
                    found = Some((
                    r.get_x(),
                    r.get_y(),
                    r.get_rotation(),
                    r.get_width(),
                    r.get_height(),
                    r.get_scale_x(),
                    r.get_scale_y(),
                    ));
                    break;
                    }
                    }
                    if found.is_some() {
                    break;
                    }
                    }
                    if let Some((rx, ry, rrot, rw, rh, rsx, rsy)) = found {
                    cfg.x = rx;
                    cfg.y = ry;
                    cfg.rotation = rrot;
                    let ref_w = (rw * rsx.abs()).max(1.0);
                    let ref_h = (rh * rsy.abs()).max(1.0);
                    cfg.scale_x = (ref_w / cfg.width.max(1.0)) * rsx.signum();
                    cfg.scale_y = (ref_h / cfg.height.max(1.0)) * rsy.signum();
                    self.update_custom_sprite_attachment(&target_slot, &cfg);
                    self.custom_sprites.insert(target_slot.clone(), cfg.clone());
                    }
                    }
                    }
                    if ui.button("Move to Front").clicked() {
                    self.set_slot_layer(&target_slot, true);
                    }
                    if ui.button("Move to Back").clicked() {
                    self.set_slot_layer(&target_slot, false);
                    }
                    if ui.button("Remove Part").clicked() {
                    remove_part = true;
                    }
                    });
                    if remove_part {
                    self.remove_custom_part(&target_slot);
                    }
                    }
                    if !parts.is_empty()
                    && ui
                    .button("Save Custom Parts & Edits to Spine Export (JSON + Atlas)")
                    .clicked()
                    {
                    match self.export_modified_spine() {
                    Ok(msg) => {
                    self.save_status = Some(msg);
                    self.last_error = None;
                    }
                    Err(e) => {
                    self.last_error = Some(e);
                    self.save_status = None;
                    }
                    }
                    }
                    });

                    ui.separator();

                    }
                    StudioTab::BoneStudio => {
                    egui::CollapsingHeader::new("🦴 Bone Modifying & Hierarchy Studio")
                    .default_open(true)
                    .show(ui, |ui| {
                    ui.checkbox(
                    &mut self.show_bone_gizmos,
                    "👁️ Show Bones in Viewport (Click any joint in viewport to select)",
                    );
                    if let Some(data) = self.data.as_ref() {
                    let bones = data.get_bones();
                    if self.selected_bone.is_empty()
                    && let Some(first) = bones.first()
                    {
                    self.selected_bone = first.get_name().to_string();
                    }

                    ui.label("Quick Select Bone:");
                    ui.horizontal_wrapped(|ui| {
                    for (icon, label, name) in [
                    ("🖐️", "Left Hand", "l_hand_C2"),
                    ("🖐️", "Right Hand", "r_hand_C2"),
                    ("👗", "Body", "body"),
                    ("👤", "Head", "head"),
                    ("🦵", "Left Leg", "l_leg"),
                    ("🦵", "Right Leg", "r_leg"),
                    ("⚓", "Root", "root"),
                    ] {
                    if bones.iter().any(|b| b.get_name() == name) {
                    let is_sel = self.selected_bone == name;
                    if ui.selectable_label(is_sel, format!("{icon} {label}")).clicked() {
                    self.selected_bone = name.to_string();
                    }
                    }
                    }
                    });

                    ui.horizontal(|ui| {
                    ui.label("Filter bone:");
                    ui.text_edit_singleline(&mut self.bone_filter);
                    if !self.bone_filter.is_empty() && ui.button("X").clicked() {
                    self.bone_filter.clear();
                    }
                    });

                    let filter = self.bone_filter.trim().to_ascii_lowercase();
                    ui.horizontal(|ui| {
                    ui.label("Bone:");
                    egui::ComboBox::from_id_salt("bone_editor_select")
                    .width(220.0)
                    .selected_text(&self.selected_bone)
                    .show_ui(ui, |ui| {
                    for b in bones {
                    let b_name = b.get_name();
                    if !filter.is_empty()
                    && !b_name.to_ascii_lowercase().contains(&filter)
                    {
                    continue;
                    }
                    let parent_label = b
                    .get_parent(data)
                    .map(|p| format!("{b_name}  (<- {})", p.get_name()))
                    .unwrap_or_else(|| format!("{b_name}  (root)"));
                    ui.selectable_value(
                    &mut self.selected_bone,
                    b_name.to_string(),
                    parent_label,
                    );
                    }
                    });
                    if !self.bone_offsets.is_empty()
                    && ui
                    .button(format!("Reset All ({})", self.bone_offsets.len()))
                    .clicked()
                    {
                    self.bone_offsets.clear();
                    }
                    });

                    if !self.selected_bone.is_empty() {
                    let cur_b_name = self.selected_bone.clone();
                    let parent_name = bones
                    .iter()
                    .find(|b| b.get_name() == cur_b_name)
                    .and_then(|b| b.get_parent(data))
                    .map(|p| p.get_name().to_string())
                    .unwrap_or_else(|| "none (root)".to_string());

                    ui.colored_label(
                    egui::Color32::from_rgb(255, 230, 100),
                    format!("🦴 Active: {cur_b_name} (Parent: {parent_name})"),
                    );

                    let mut offset = self
                    .bone_offsets
                    .get(&cur_b_name)
                    .copied()
                    .unwrap_or_default();
                    let mut changed = false;

                    egui::Grid::new("bone_offset_sliders")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                    ui.label("Offset X");
                    ui.horizontal(|ui| {
                    changed |= ui
                    .add(
                    egui::Slider::new(&mut offset.dx, -500.0..=500.0)
                    .step_by(0.5),
                    )
                    .changed();
                    if ui.button("-5").clicked() { offset.dx -= 5.0; changed = true; }
                    if ui.button("+5").clicked() { offset.dx += 5.0; changed = true; }
                    if ui.button("0").clicked() { offset.dx = 0.0; changed = true; }
                    });
                    ui.end_row();

                    ui.label("Offset Y");
                    ui.horizontal(|ui| {
                    changed |= ui
                    .add(
                    egui::Slider::new(&mut offset.dy, -500.0..=500.0)
                    .step_by(0.5),
                    )
                    .changed();
                    if ui.button("-5").clicked() { offset.dy -= 5.0; changed = true; }
                    if ui.button("+5").clicked() { offset.dy += 5.0; changed = true; }
                    if ui.button("0").clicked() { offset.dy = 0.0; changed = true; }
                    });
                    ui.end_row();

                    ui.label("Rotate (deg)");
                    ui.horizontal(|ui| {
                    changed |= ui
                    .add(
                    egui::Slider::new(&mut offset.drot, -180.0..=180.0)
                    .step_by(0.5),
                    )
                    .changed();
                    if ui.button("-45°").clicked() { offset.drot -= 45.0; changed = true; }
                    if ui.button("+45°").clicked() { offset.drot += 45.0; changed = true; }
                    if ui.button("0°").clicked() { offset.drot = 0.0; changed = true; }
                    });
                    ui.end_row();

                    ui.label("Scale X");
                    ui.horizontal(|ui| {
                    changed |= ui
                    .add(
                    egui::Slider::new(&mut offset.sx, -3.0..=3.0)
                    .step_by(0.01),
                    )
                    .changed();
                    if ui.button("Flip X").clicked() { offset.sx = -offset.sx; changed = true; }
                    if ui.button("1.0").clicked() { offset.sx = 1.0; changed = true; }
                    });
                    ui.end_row();

                    ui.label("Scale Y");
                    ui.horizontal(|ui| {
                    changed |= ui
                    .add(
                    egui::Slider::new(&mut offset.sy, -3.0..=3.0)
                    .step_by(0.01),
                    )
                    .changed();
                    if ui.button("Flip Y").clicked() { offset.sy = -offset.sy; changed = true; }
                    if ui.button("1.0").clicked() { offset.sy = 1.0; changed = true; }
                    });
                    ui.end_row();

                    ui.label("Bone Length");
                    ui.horizontal(|ui| {
                    let cur_len = if offset.length > 0.0 {
                    offset.length
                    } else {
                    bones
                    .iter()
                    .find(|b| b.get_name() == cur_b_name)
                    .map(|b| b.get_length())
                    .unwrap_or(0.0)
                    };
                    let mut temp_len = cur_len;
                    if ui
                    .add(
                    egui::Slider::new(&mut temp_len, 0.0..=400.0)
                    .step_by(1.0),
                    )
                    .changed()
                    {
                    offset.length = temp_len;
                    changed = true;
                    }
                    });
                    ui.end_row();
                    });

                    ui.horizontal_wrapped(|ui| {
                    if ui.button("Reset This Bone").clicked() {
                    offset = BoneOffset::default();
                    changed = true;
                    }

                    let mirror_target = if cur_b_name == "l_hand_C2" {
                    Some("r_hand_C2".to_string())
                    } else if cur_b_name == "r_hand_C2" {
                    Some("l_hand_C2".to_string())
                    } else if cur_b_name.starts_with("l_") {
                    let cand = format!("r_{}", &cur_b_name[2..]);
                    if bones.iter().any(|b| b.get_name() == cand) { Some(cand) } else { None }
                    } else if cur_b_name.starts_with("r_") {
                    let cand = format!("l_{}", &cur_b_name[2..]);
                    if bones.iter().any(|b| b.get_name() == cand) { Some(cand) } else { None }
                    } else {
                    None
                    };

                    if let Some(target_b) = mirror_target {
                    if ui.button(format!("🪞 Mirror to '{target_b}'")).clicked() {
                    let mut mir_off = offset;
                    mir_off.dx = -mir_off.dx;
                    mir_off.drot = -mir_off.drot;
                    self.bone_offsets.insert(target_b.clone(), mir_off);
                    self.save_status = Some(format!("Mirrored transforms to '{target_b}'"));
                    }
                    }
                    });

                    if changed {
                    if offset.is_identity() {
                    self.bone_offsets.remove(&cur_b_name);
                    } else {
                    self.bone_offsets.insert(cur_b_name.clone(), offset);
                    }
                    }

                    ui.separator();
                    ui.strong("Add Child Bone / Reparent:");
                    ui.horizontal(|ui| {
                    ui.label("Add Bone:");
                    ui.add(
                    egui::TextEdit::singleline(&mut self.new_bone_name)
                    .desired_width(100.0)
                    .hint_text("prop_bone"),
                    );
                    ui.label("on Parent:");
                    egui::ComboBox::from_id_salt("new_bone_parent_combo")
                    .width(110.0)
                    .selected_text(&self.new_bone_parent)
                    .show_ui(ui, |ui| {
                    for b in bones {
                    let bn = b.get_name().to_string();
                    ui.selectable_value(&mut self.new_bone_parent, bn.clone(), bn);
                    }
                    });
                    if ui.button("+ Add Bone").clicked() {
                    let b_name = self.new_bone_name.trim().to_string();
                    let p_name = self.new_bone_parent.trim().to_string();
                    if !b_name.is_empty() && !p_name.is_empty() {
                    if !self.created_bones.iter().any(|(n, _)| n == &b_name) {
                    self.created_bones.push((b_name.clone(), p_name.clone()));
                    self.selected_bone = b_name.clone();
                    self.save_status = Some(format!(
                    "Added bone '{b_name}' on parent '{p_name}'. Click 'Save & Recompile' to export!"
                    ));
                    }
                    }
                    }
                    });

                    ui.horizontal(|ui| {
                    ui.label("Reparent:");
                    let mut cur_parent = self
                    .bone_parent_overrides
                    .get(&cur_b_name)
                    .cloned()
                    .unwrap_or_else(|| parent_name.clone());
                    let orig_parent = cur_parent.clone();
                    egui::ComboBox::from_id_salt("reparent_combo")
                    .width(130.0)
                    .selected_text(&cur_parent)
                    .show_ui(ui, |ui| {
                    for b in bones {
                    let bn = b.get_name().to_string();
                    if bn != cur_b_name {
                    ui.selectable_value(&mut cur_parent, bn.clone(), bn);
                    }
                    }
                    });
                    if cur_parent != orig_parent {
                    self.bone_parent_overrides.insert(cur_b_name, cur_parent);
                    }
                    });
                    }
                    }
                    });

                    ui.separator();

                    }
                    StudioTab::SkinBuilder => {
                    egui::CollapsingHeader::new("Skin Builder (Save Outfit as Character Skin)")
                    .default_open(true)
                    .show(ui, |ui| {
                    ui.label(
                    "Bakes the active skin + slot overrides + custom PNG parts into a new skin.",
                    );
                    ui.horizontal(|ui| {
                    ui.label("New skin name:");
                    ui.add(
                    egui::TextEdit::singleline(&mut self.new_skin_name)
                    .desired_width(180.0)
                    .hint_text("Custom_Outfit_1"),
                    );
                    });
                    if ui.button("+ Save Current Outfit as New Skin").clicked() {
                    let name = self.new_skin_name.clone();
                    match self.save_current_outfit_as_skin(&name) {
                    Ok(msg) => {
                    self.save_status = Some(msg);
                    self.last_error = None;
                    }
                    Err(e) => {
                    self.last_error = Some(e);
                    self.save_status = None;
                    }
                    }
                    }
                    if !self.custom_skins.is_empty() {
                    ui.label("Custom skins (click to switch; Save & Recompile persists them):");
                    let names: Vec<String> =
                    self.custom_skins.iter().map(|r| r.name.clone()).collect();
                    let mut picked: Option<String> = None;
                    ui.horizontal_wrapped(|ui| {
                    for n in &names {
                    if ui
                    .selectable_label(self.ui.skin == *n, n)
                    .clicked()
                    {
                    picked = Some(n.clone());
                    }
                    }
                    });
                    if let Some(n) = picked {
                    self.ui.skin = n.clone();
                    if let Some(sk) = self.skeleton.as_mut() {
                    sk.set_skin(Some(&n));
                    sk.setup_pose();
                    }
                    self.activate_custom_skin_outfit(&n);
                    }
                    }
                    });

                    ui.separator();

                    }
                    StudioTab::LayerOrder => {
                    egui::CollapsingHeader::new("Draw Order / Layer Rearranger")
                    .default_open(false)
                    .show(ui, |ui| {
                    if let Some(data) = self.data.as_ref() {
                    let slot_count = data.get_slots().len();
                    ui.horizontal(|ui| {
                    ui.label("Filter layer:");
                    ui.text_edit_singleline(&mut self.draw_order_filter);
                    if !self.draw_order_filter.is_empty() && ui.button("X").clicked() {
                    self.draw_order_filter.clear();
                    }
                    if self.custom_draw_order.is_some()
                    && ui.button("Reset Draw Order").clicked()
                    {
                    self.custom_draw_order = None;
                    }
                    });

                    let current_order: Vec<usize> = self
                    .custom_draw_order
                    .clone()
                    .or_else(|| {
                    self.skeleton
                    .as_ref()
                    .map(|sk| sk.get_draw_order().to_vec())
                    })
                    .unwrap_or_else(|| (0..slot_count).collect());

                    let filter = self.draw_order_filter.trim().to_ascii_lowercase();
                    let mut next_order = current_order.clone();
                    let mut order_changed = false;

                    egui::ScrollArea::vertical()
                    .id_salt("draw_order_scroll")
                    .max_height(240.0)
                    .show(ui, |ui| {
                    // Display from Front (top of draw order, end of slice) to Back (start of slice)
                    for pos in (0..current_order.len()).rev() {
                    let slot_idx = current_order[pos];
                    let slot_name = data
                    .get_slots()
                    .get(slot_idx)
                    .map(|s| s.get_name())
                    .unwrap_or("?");
                    if !filter.is_empty()
                    && !slot_name.to_ascii_lowercase().contains(&filter)
                    {
                    continue;
                    }
                    ui.horizontal(|ui| {
                    ui.label(format!("#{pos:02}"));
                    if ui
                    .add_enabled(
                    pos + 1 < current_order.len(),
                    egui::Button::new("Front ^"),
                    )
                    .clicked()
                    {
                    next_order.swap(pos, pos + 1);
                    order_changed = true;
                    }
                    if ui
                    .add_enabled(pos > 0, egui::Button::new("Back v"))
                    .clicked()
                    {
                    next_order.swap(pos, pos - 1);
                    order_changed = true;
                    }
                    if ui
                    .add_enabled(
                    pos + 1 < current_order.len(),
                    egui::Button::new("Top"),
                    )
                    .clicked()
                    {
                    let item = next_order.remove(pos);
                    next_order.push(item);
                    order_changed = true;
                    }
                    if ui
                    .add_enabled(pos > 0, egui::Button::new("Bottom"))
                    .clicked()
                    {
                    let item = next_order.remove(pos);
                    next_order.insert(0, item);
                    order_changed = true;
                    }
                    ui.label(slot_name);
                    });
                    }
                    });

                    if order_changed {
                    self.custom_draw_order = Some(next_order);
                    }
                    }
                    });

                    }

                    StudioTab::ImportExport => {
                        egui::CollapsingHeader::new("📥 Spine Character Importer (.json / .skel / folder)")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.label("Import any Spine export from anywhere on your computer without having to copy it beforehand.");
                                ui.colored_label(
                                    egui::Color32::from_rgb(100, 200, 255),
                                    "🛡️ Non-destructive: Copies the files into the examples folder — your original files are NEVER overwritten!",
                                );
                                ui.add_space(4.0);

                                ui.horizontal(|ui| {
                                    if ui.button("📄 Browse Spine File (.json / .skel / .atlas)...").clicked() {
                                        self.pick_and_import_file();
                                    }
                                    if ui.button("📁 Browse Character Folder...").clicked() {
                                        self.pick_and_import_folder();
                                    }
                                });

                                ui.add_space(6.0);
                                ui.label("Quick Import from SquishyDumpling:");
                                let chars = squishy_characters();
                                if chars.is_empty() {
                                    ui.weak(format!("(no character folders found in {SQUISHY_SKIN_DIR})"));
                                }
                                let mut quick_pick: Option<String> = None;
                                ui.horizontal_wrapped(|ui| {
                                    for c in &chars {
                                        if ui.button(c).clicked() {
                                            quick_pick = Some(c.clone());
                                        }
                                    }
                                });
                                if let Some(c) = quick_pick {
                                    let name: String = c
                                        .chars()
                                        .filter(|ch| ch.is_ascii_alphanumeric())
                                        .map(|ch| ch.to_ascii_lowercase())
                                        .collect();
                                    let already = self.examples.contains(&name)
                                        && pick_export_files(&name, false).is_ok();
                                    if already {
                                        self.load_imported_example(&name, format!("Loaded '{name}'"));
                                    } else {
                                        self.import_and_load(&Path::new(SQUISHY_SKIN_DIR).join(&c));
                                    }
                                }
                                if let Some(msg) = self.import_status.as_deref() {
                                    ui.colored_label(egui::Color32::LIGHT_GREEN, msg);
                                }
                            });
                        ui.separator();

                        ui.separator();
                        ui.strong("💾 Save & Export Modified Spine:");
                        ui.label("Bakes all bone modifications, new bones, part sprite transforms, slot re-wirings, and custom parts directly into the Spine JSON + Atlas files in the example directory.");
                        ui.horizontal(|ui| {
                            if ui.button("💾 Save & Recompile Spine Export (JSON + Atlas)").clicked() {
                                match self.export_modified_spine() {
                                    Ok(msg) => {
                                        self.save_status = Some(msg);
                                        self.last_error = None;
                                    }
                                    Err(e) => {
                                        self.last_error = Some(e);
                                    }
                                }
                            }
                            if ui.button("📂 Open Export Folder").on_hover_text("Open current export directory in Windows Explorer").clicked() {
                                if let Some(parent) = self.json_path.parent() {
                                    let _ = std::process::Command::new("explorer").arg(parent).spawn();
                                }
                            }
                        });
                        if let Some(msg) = self.save_status.as_deref() {
                            ui.colored_label(egui::Color32::LIGHT_GREEN, msg);
                        }
                    }

                }

                if let Some(err) = self.last_error.as_deref() {
                    ui.separator();
                    ui.colored_label(egui::Color32::LIGHT_RED, err);
                }
            });

        if self.show_bone_gizmos {
            self.draw_bone_gizmos(ctx);
        }

        if self.needs_fit {
            self.needs_fit = false;
            self.recompute_fit();
        }
    }

    fn draw_bone_gizmos(&mut self, ctx: &egui::Context) {
        let (Some(skeleton), Some(data), Some(config), Some(window)) = (
            self.skeleton.as_ref(),
            self.data.as_ref(),
            self.config.as_ref(),
            self.window.as_ref(),
        ) else {
            return;
        };

        let ppp = window.scale_factor() as f32;
        let vw = config.width.max(1) as f32;
        let vh = config.height.max(1) as f32;

        let matrix = if let Some(bounds) = self.initial_bounds {
            clip_from_world_fit_bounds(bounds, config.width, config.height, self.fit_margin)
        } else {
            let sx = 2.0 / vw;
            let sy = 2.0 / vh;
            [
                [sx, 0.0, 0.0, 0.0],
                [0.0, sy, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        };

        let world_to_screen = |wx: f32, wy: f32| -> egui::Pos2 {
            let clip_x = matrix[0][0] * wx + matrix[1][0] * wy + matrix[3][0];
            let clip_y = matrix[0][1] * wx + matrix[1][1] * wy + matrix[3][1];
            let sx = (clip_x + 1.0) * 0.5 * (vw / ppp);
            let sy = (1.0 - clip_y) * 0.5 * (vh / ppp);
            egui::pos2(sx, sy)
        };

        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Background,
            egui::Id::new("bone_gizmos_layer"),
        ));

        let bones = skeleton.get_bones();
        let bone_defs = data.get_bones();

        let pointer_pos = ctx.input(|i| i.pointer.hover_pos());
        let clicked = ctx.input(|i| i.pointer.primary_clicked());
        let mut clicked_bone = None;

        for (i, bone) in bones.iter().enumerate() {
            if !bone.is_active() {
                continue;
            }
            let Some(def) = bone_defs.get(i) else {
                continue;
            };
            let b_name = def.get_name();
            let is_selected = b_name == self.selected_bone;
            let start = world_to_screen(bone.get_world_x(), bone.get_world_y());
            let off_len = self.bone_offsets.get(b_name).map(|o| o.length).unwrap_or(0.0);
            let len = if off_len > 0.0 { off_len } else { def.get_length() };

            if let Some(p) = pointer_pos {
                if (p - start).length() < 14.0 {
                    painter.text(
                        start + egui::vec2(10.0, -10.0),
                        egui::Align2::LEFT_BOTTOM,
                        format!("🦴 {b_name} (Click to Select)"),
                        egui::FontId::proportional(13.0),
                        egui::Color32::from_rgb(100, 240, 255),
                    );
                    if clicked {
                        clicked_bone = Some(b_name.to_string());
                    }
                }
            }

            if len > 1.0 {
                let end_wx = bone.get_a() * len + bone.get_world_x();
                let end_wy = bone.get_c() * len + bone.get_world_y();
                let end = world_to_screen(end_wx, end_wy);
                let stroke = if is_selected {
                    egui::Stroke::new(3.5, egui::Color32::from_rgb(255, 225, 40))
                } else {
                    egui::Stroke::new(1.5, egui::Color32::from_rgba_unmultiplied(0, 230, 255, 170))
                };
                painter.line_segment([start, end], stroke);
            } else if let Some(parent) = bone.get_parent(skeleton) {
                let p_pos = world_to_screen(parent.get_world_x(), parent.get_world_y());
                painter.line_segment(
                    [p_pos, start],
                    egui::Stroke::new(
                        1.0,
                        egui::Color32::from_rgba_unmultiplied(160, 160, 160, 90),
                    ),
                );
            }

            let dot_color = if is_selected {
                egui::Color32::from_rgb(255, 60, 60)
            } else {
                egui::Color32::from_rgb(50, 255, 120)
            };
            painter.circle_filled(start, if is_selected { 6.0 } else { 3.0 }, dot_color);
            if is_selected {
                let axis_len = 25.0;
                let ax_wx = bone.get_world_x() + bone.get_a() * axis_len;
                let ax_wy = bone.get_world_y() + bone.get_c() * axis_len;
                let ay_wx = bone.get_world_x() + bone.get_b() * axis_len;
                let ay_wy = bone.get_world_y() + bone.get_d() * axis_len;
                let ax_screen = world_to_screen(ax_wx, ax_wy);
                let ay_screen = world_to_screen(ay_wx, ay_wy);
                painter.line_segment([start, ax_screen], egui::Stroke::new(2.5, egui::Color32::RED));
                painter.line_segment([start, ay_screen], egui::Stroke::new(2.5, egui::Color32::GREEN));

                painter.text(
                    start + egui::vec2(8.0, -8.0),
                    egui::Align2::LEFT_BOTTOM,
                    format!("🦴 {b_name}"),
                    egui::FontId::proportional(14.0),
                    egui::Color32::from_rgb(255, 240, 50),
                );
            }
        }

        if let Some(b) = clicked_bone {
            self.selected_bone = b;
        }
    }

    fn apply_pending_reload(&mut self) {
        let Some(next) = self.pending_reload.take() else {
            return;
        };
        let (Some(window), Some(device), Some(queue), Some(renderer), Some(config)) = (
            self.window.as_ref(),
            self.device.as_ref(),
            self.queue.as_ref(),
            self.renderer.as_mut(),
            self.config.as_ref(),
        ) else {
            self.pending_reload = Some(next);
            return;
        };

        let example = sanitize_example_arg(&next.example);
        let (json_path, atlas_path) = match pick_export_files(&example, next.prefer_pma_atlas) {
            Ok(v) => v,
            Err(e) => {
                self.last_error = Some(e);
                return;
            }
        };

        let atlas_dir = match atlas_path.parent() {
            Some(p) => p,
            None => {
                self.last_error = Some(format!("atlas path has no parent dir: {atlas_path:?}"));
                return;
            }
        };

        let json = match std::fs::read_to_string(&json_path) {
            Ok(s) => s,
            Err(e) => {
                self.last_error = Some(format!("read json {json_path:?}: {e}"));
                return;
            }
        };
        let data = match SkeletonData::from_json_str(&json) {
            Ok(d) => d,
            Err(e) => {
                self.last_error = Some(format!("parse json {json_path:?}: {e}"));
                return;
            }
        };

        let atlas_text = match std::fs::read_to_string(&atlas_path) {
            Ok(s) => s,
            Err(e) => {
                self.last_error = Some(format!("read atlas {atlas_path:?}: {e}"));
                return;
            }
        };
        let atlas = match atlas_text.parse::<Atlas>() {
            Ok(a) => a,
            Err(e) => {
                self.last_error = Some(format!("parse atlas {atlas_path:?}: {e}"));
                return;
            }
        };

        let mut skeleton = Skeleton::new(data.clone());
        let chosen_skin = {
            let requested = next.skin.trim();
            if requested == SKIN_NONE_SENTINEL {
                None
            } else if !requested.is_empty() && data.find_skin(requested).is_some() {
                Some(requested.to_string())
            } else {
                choose_default_skin(&example, &data)
            }
        };
        if let Some(skin_name) = chosen_skin.as_deref() {
            skeleton.set_skin(Some(skin_name));
        }
        skeleton.setup_pose();
        skeleton.update_world_transform_with_physics(spine2d::Physics::None);

        let mut state = AnimationState::new(AnimationStateData::new(data.clone()));
        let chosen_animation = {
            let requested = next.animation.trim();
            if !requested.is_empty() && data.find_animation(requested).is_some() {
                Some(requested.to_string())
            } else {
                choose_default_animation(&data)
            }
        };
        let Some(chosen_animation) = chosen_animation else {
            self.last_error = Some("No animations in skeleton data.".to_string());
            return;
        };
        state.set_animation(0, chosen_animation.as_str(), true);

        skeleton.setup_pose();
        state.apply(&mut skeleton);
        Self::apply_runtime_edits(
            &mut skeleton,
            &data,
            &self.slot_overrides,
            &self.slot_bone_overrides,
            &self.bone_offsets,
            self.custom_draw_order.as_ref(),
        );
        skeleton.update_world_transform_with_physics(spine2d::Physics::None);

        let mut textures = HashMapTextureProvider {
            bind_groups: HashMap::new(),
        };
        for page in atlas.get_pages() {
            let image_path = atlas_dir.join(&page.name);
            let (w, h, pixels) = match load_png_rgba8(&image_path) {
                Ok(v) => v,
                Err(e) => {
                    self.last_error = Some(e);
                    return;
                }
            };

            let sampler = create_sampler_for_atlas_page(device, page);
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("spine2d page texture"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                texture.as_image_copy(),
                &pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * w),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let bind_group = create_texture_bind_group(
                device,
                renderer.texture_bind_group_layout(),
                &view,
                &sampler,
            );
            textures.bind_groups.insert(page.name.clone(), bind_group);
        }

        self.draw_list.clear();
        spine2d::append_draw_list_with_atlas(&mut self.draw_list, &skeleton, &atlas);
        self.initial_bounds = bounds_from_draw_list(&self.draw_list);
        self.example = example;
        self.animation = Some(chosen_animation);
        self.speed = next.speed;
        self.fit_margin = next.margin;
        self.prefer_pma_atlas = next.prefer_pma_atlas;
        self.json_path = json_path;
        self.atlas_path = atlas_path;
        self.textures = textures;
        self.atlas = Some(atlas);
        let data_ref = data.clone();
        self.data = Some(data);
        self.skeleton = Some(skeleton);
        self.state = Some(state);
        self.last_frame = Some(Instant::now());
        self.last_error = None;

        self.ui.example = self.example.clone();
        // If the chosen animation doesn't exist in the newly loaded skeleton, fall back to auto.
        self.ui.animation = self
            .animation
            .clone()
            .filter(|name| data_ref.find_animation(name).is_some())
            .unwrap_or_default();
        self.ui.skin = chosen_skin.unwrap_or_else(|| SKIN_NONE_SENTINEL.to_string());
        self.ui.prefer_pma_atlas = self.prefer_pma_atlas;
        self.ui.speed = self.speed;
        self.ui.margin = self.fit_margin;

        window.set_title(&format!("spine2d-wgpu: {}", self.example));
        if let Some(bounds) = self.initial_bounds {
            renderer.update_globals_matrix(
                queue,
                clip_from_world_fit_bounds(bounds, config.width, config.height, self.fit_margin),
            );
        } else {
            renderer.update_globals_ortho_centered(
                queue,
                config.width as f32,
                config.height as f32,
            );
        }
        renderer.upload(device, queue, &self.draw_list);
        window.request_redraw();

        // Re-apply any unsaved created slots and custom PNG sprites after a reload
        let saved_created_slots = self.created_slots.clone();
        for (s_name, b_name) in saved_created_slots {
            let _ = self.create_new_slot(&s_name, &b_name);
        }
        let saved_custom_sprites: Vec<(String, CustomSpriteConfig)> = self
            .custom_sprites
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        for (slot_name, cfg) in saved_custom_sprites {
            if self
                .upload_custom_texture(&slot_name, Path::new(&cfg.source_path))
                .is_ok()
            {
                self.update_custom_sprite_attachment(&slot_name, &cfg);
            }
        }
        // Re-register unsaved custom skins (built via the Skin Builder) and re-select if requested.
        self.reapply_custom_skins(next.skin.trim());
    }
}

fn choose_default_animation(data: &SkeletonData) -> Option<String> {
    // Prefer more "stable" animations to avoid defaulting to something very fast (e.g. run).
    for name in ["idle", "walk", "run"] {
        if data.find_animation(name).is_some() {
            return Some(name.to_string());
        }
    }
    data.get_animations()
        .first()
        .map(|a| a.get_name().to_string())
}

fn choose_default_skin(example: &str, data: &SkeletonData) -> Option<String> {
    // Match the web demo defaults for common examples.
    let recommended = match example {
        "goblins" => Some("goblin"),
        "mix-and-match" => Some("full-skins/girl-blue-cape"),
        "chibi-stickers" => Some("spineboy"),
        "mira" => Some("Mira"),
        _ => None,
    };
    if let Some(name) = recommended
        && data.find_skin(name).is_some()
    {
        return Some(name.to_string());
    }

    if data.find_skin("default").is_some() {
        return Some("default".to_string());
    }

    // Fallback: pick the skin with the most attachments.
    let mut best: Option<(&str, usize)> = None;
    for skin in data.get_skins() {
        let count = skin.get_attachments().count();
        if count == 0 {
            continue;
        }
        match best {
            None => best = Some((skin.get_name(), count)),
            Some((_, best_count)) if count > best_count => {
                best = Some((skin.get_name(), count));
            }
            _ => {}
        }
    }
    best.map(|(n, _)| n.to_string())
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        self.examples = available_examples();

        let atlas_dir = self.atlas_path.parent().expect("atlas parent dir");

        let json = match std::fs::read_to_string(&self.json_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("read json {:?}: {e}", self.json_path);
                event_loop.exit();
                return;
            }
        };
        let data = match SkeletonData::from_json_str(&json) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("parse json {:?}: {e}", self.json_path);
                event_loop.exit();
                return;
            }
        };

        let atlas_text = match std::fs::read_to_string(&self.atlas_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("read atlas {:?}: {e}", self.atlas_path);
                event_loop.exit();
                return;
            }
        };
        let atlas = match atlas_text.parse::<Atlas>() {
            Ok(a) => a,
            Err(e) => {
                eprintln!("parse atlas {:?}: {e}", self.atlas_path);
                event_loop.exit();
                return;
            }
        };

        let mut skeleton = Skeleton::new(data.clone());
        let chosen_skin = choose_default_skin(&self.example, &data);
        if let Some(skin_name) = chosen_skin.as_deref() {
            skeleton.set_skin(Some(skin_name));
        }
        skeleton.setup_pose();
        skeleton.update_world_transform_with_physics(spine2d::Physics::None);

        let mut state = AnimationState::new(AnimationStateData::new(data.clone()));
        let chosen_animation = if let Some(anim_name) = self.animation.as_deref() {
            Some(anim_name.to_string())
        } else {
            // Prefer more "stable" animations to avoid defaulting to something very fast (e.g. run).
            let preferred = ["idle", "walk", "run"];
            let mut found = None;
            for name in preferred {
                if data.find_animation(name).is_some() {
                    found = Some(name.to_string());
                    break;
                }
            }
            found.or_else(|| {
                data.get_animations()
                    .first()
                    .map(|a| a.get_name().to_string())
            })
        };

        let Some(chosen_animation) = chosen_animation else {
            eprintln!("No animations in skeleton data.");
            event_loop.exit();
            return;
        };

        state.set_animation(0, chosen_animation.as_str(), true);

        // IMPORTANT: Our current runtime path assumes a clean base pose each frame
        // (otherwise constraints/local transforms can accumulate and look like "no keyframes").
        skeleton.setup_pose();
        state.apply(&mut skeleton);
        skeleton.update_world_transform_with_physics(spine2d::Physics::None);

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title(format!("spine2d-wgpu: {}", self.example)),
                )
                .unwrap(),
        );

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
        }))
        .unwrap();

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: Default::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: Default::default(),
        }))
        .unwrap();

        let size = window.inner_size().max(PhysicalSize::new(1, 1));
        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let mut renderer = SpineRenderer::new(&device, config.format);

        let mut textures = HashMapTextureProvider {
            bind_groups: HashMap::new(),
        };
        for page in atlas.get_pages() {
            let image_path = atlas_dir.join(&page.name);
            let (w, h, pixels) = match load_png_rgba8(&image_path) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("{e}");
                    event_loop.exit();
                    return;
                }
            };

            let sampler = create_sampler_for_atlas_page(&device, page);
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("spine2d page texture"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                texture.as_image_copy(),
                &pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * w),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let bind_group = create_texture_bind_group(
                &device,
                renderer.texture_bind_group_layout(),
                &view,
                &sampler,
            );
            textures.bind_groups.insert(page.name.clone(), bind_group);
        }

        self.draw_list = spine2d::build_draw_list_with_atlas(&skeleton, &atlas);
        self.initial_bounds = bounds_from_draw_list(&self.draw_list);
        if let Some(bounds) = self.initial_bounds {
            renderer.update_globals_matrix(
                &queue,
                clip_from_world_fit_bounds(bounds, config.width, config.height, self.fit_margin),
            );
        } else {
            renderer.update_globals_ortho_centered(
                &queue,
                config.width as f32,
                config.height as f32,
            );
        }
        renderer.upload(&device, &queue, &self.draw_list);

        let egui_state = egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            window.theme(),
            None,
        );
        let egui_renderer =
            EguiRenderer::new(&device, config.format, EguiRendererOptions::default());

        window.request_redraw();
        self.window = Some(window);
        self.surface = Some(surface);
        self.device = Some(device);
        self.queue = Some(queue);
        self.config = Some(config);
        self.renderer = Some(renderer);
        self.textures = textures;
        self.atlas = Some(atlas);
        self.data = Some(data);
        self.skeleton = Some(skeleton);
        self.state = Some(state);
        self.egui_state = Some(egui_state);
        self.egui_renderer = Some(egui_renderer);
        self.ui = UiSelection {
            example: self.example.clone(),
            animation: chosen_animation,
            skin: chosen_skin.unwrap_or_else(|| SKIN_NONE_SENTINEL.to_string()),
            prefer_pma_atlas: self.prefer_pma_atlas,
            speed: self.speed,
            margin: self.fit_margin,
        };
        self.last_frame = Some(Instant::now());
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(window) = self.window.clone() else {
            return;
        };
        let window_ref = window.as_ref();

        if let Some(egui_state) = self.egui_state.as_mut() {
            let response = egui_state.on_window_event(window_ref, &event);
            if response.repaint {
                window_ref.request_redraw();
            }
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                let Some(surface) = self.surface.as_ref() else {
                    return;
                };
                let Some(device) = self.device.as_ref() else {
                    return;
                };
                let Some(queue) = self.queue.as_ref() else {
                    return;
                };
                let Some(config) = self.config.as_mut() else {
                    return;
                };
                config.width = size.width.max(1);
                config.height = size.height.max(1);
                surface.configure(device, config);
                if let Some(renderer) = self.renderer.as_ref() {
                    if let Some(bounds) = self.initial_bounds {
                        renderer.update_globals_matrix(
                            queue,
                            clip_from_world_fit_bounds(
                                bounds,
                                config.width,
                                config.height,
                                self.fit_margin,
                            ),
                        );
                    } else {
                        renderer.update_globals_ortho_centered(
                            queue,
                            config.width as f32,
                            config.height as f32,
                        );
                    }
                }
                window_ref.request_redraw();
            }
            WindowEvent::Occluded(false) => window_ref.request_redraw(),
            WindowEvent::RedrawRequested => {
                self.apply_pending_reload();

                let now = Instant::now();
                let mut dt = self
                    .last_frame
                    .replace(now)
                    .map(|t| (now - t).as_secs_f32())
                    .unwrap_or(0.0)
                    .min(1.0 / 15.0);
                dt = (dt * self.speed).max(0.0);

                if self.debug_time {
                    self.debug_accum += dt;
                    self.debug_frames += 1;
                    if self.debug_accum >= 1.0 {
                        let fps = self.debug_frames as f32 / self.debug_accum.max(1.0e-6);
                        eprintln!("dt={:.4} speed={:.2} fps~{:.1}", dt, self.speed, fps);
                        self.debug_accum = 0.0;
                        self.debug_frames = 0;
                    }
                }
                let Some(raw_input) = self
                    .egui_state
                    .as_mut()
                    .map(|s| s.take_egui_input(window_ref))
                else {
                    return;
                };
                let egui_ctx = self.egui_ctx.clone();
                let egui::FullOutput {
                    platform_output,
                    textures_delta,
                    shapes,
                    pixels_per_point,
                    ..
                } = egui_ctx.run_ui(raw_input, |ui| self.ui_panel(ui.ctx()));
                if let Some(egui_state) = self.egui_state.as_mut() {
                    egui_state.handle_platform_output(window_ref, platform_output);
                }
                let clipped_primitives = egui_ctx.tessellate(shapes, pixels_per_point);

                self.apply_pending_custom_png();

                let Some(surface) = self.surface.as_ref() else {
                    return;
                };
                let Some(device) = self.device.as_ref() else {
                    return;
                };
                let Some(queue) = self.queue.as_ref() else {
                    return;
                };
                let Some(config) = self.config.as_ref() else {
                    return;
                };
                let Some(renderer) = self.renderer.as_mut() else {
                    return;
                };
                let Some(egui_renderer) = self.egui_renderer.as_mut() else {
                    return;
                };

                if let (Some(atlas), Some(data), Some(skeleton), Some(state)) = (
                    self.atlas.as_ref(),
                    self.data.as_ref(),
                    self.skeleton.as_mut(),
                    self.state.as_mut(),
                ) {
                    if !self.paused {
                        state.update(dt);
                    }
                    skeleton.setup_pose();
                    state.apply(skeleton);
                    Self::apply_runtime_edits(
                        skeleton,
                        data,
                        &self.slot_overrides,
                        &self.slot_bone_overrides,
                        &self.bone_offsets,
                        self.custom_draw_order.as_ref(),
                    );
                    skeleton.update_world_transform_with_physics(spine2d::Physics::None);

                    self.draw_list.clear();
                    spine2d::append_draw_list_with_atlas(&mut self.draw_list, skeleton, atlas);
                    renderer.upload(device, queue, &self.draw_list);
                }

                for (id, image_delta) in &textures_delta.set {
                    egui_renderer.update_texture(device, queue, *id, image_delta);
                }
                for id in &textures_delta.free {
                    egui_renderer.free_texture(id);
                }

                let (frame, reconfigure_surface) = match surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(frame) => (frame, false),
                    wgpu::CurrentSurfaceTexture::Suboptimal(frame) => (frame, true),
                    wgpu::CurrentSurfaceTexture::Outdated => {
                        surface.configure(device, config);
                        window_ref.request_redraw();
                        return;
                    }
                    wgpu::CurrentSurfaceTexture::Timeout => {
                        window_ref.request_redraw();
                        return;
                    }
                    wgpu::CurrentSurfaceTexture::Occluded => return,
                    wgpu::CurrentSurfaceTexture::Lost => {
                        eprintln!("surface lost");
                        event_loop.exit();
                        return;
                    }
                    wgpu::CurrentSurfaceTexture::Validation => {
                        eprintln!("surface validation failed");
                        event_loop.exit();
                        return;
                    }
                };
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encoder"),
                });

                let egui_screen_descriptor = EguiScreenDescriptor {
                    size_in_pixels: [config.width, config.height],
                    pixels_per_point: window.scale_factor() as f32,
                };
                let callback_cmds = egui_renderer.update_buffers(
                    device,
                    queue,
                    &mut encoder,
                    &clipped_primitives,
                    &egui_screen_descriptor,
                );

                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("render pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    renderer.render(&mut pass, &self.draw_list, &self.textures);
                }

                {
                    let desc = wgpu::RenderPassDescriptor {
                        label: Some("egui pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    };
                    let mut pass = encoder.begin_render_pass(&desc).forget_lifetime();
                    egui_renderer.render(&mut pass, &clipped_primitives, &egui_screen_descriptor);
                }

                queue.submit(
                    callback_cmds
                        .into_iter()
                        .chain(std::iter::once(encoder.finish())),
                );
                present_surface_texture(queue, frame);
                if reconfigure_surface {
                    surface.configure(device, config);
                }

                window_ref.request_redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let raw_example = args.next().unwrap_or_else(|| "spineboy".to_string());
    let mut animation: Option<String> = None;
    let mut speed: f32 = 1.0;
    let mut fit_margin: f32 = 0.9;
    let mut prefer_pma_atlas = true;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                eprintln!(
                    "Usage: spine_runtimes <example> [animation] [--speed <factor>]\n\
Examples:\n\
  spine_runtimes spineboy\n\
  spine_runtimes spineboy run\n\
  spine_runtimes spineboy run --speed 0.7\n\
  spine_runtimes spineboy run --margin 0.75\n\
  spine_runtimes spineboy run --no-pma\n\
\n\
Notes:\n\
  - If animation is omitted, it prefers idle/walk/run, else the first animation in JSON.\n\
  - Use `python3 ./scripts/fetch_spine_runtimes_examples.py --mode export --scope tests` to fetch assets."
                );
                return;
            }
            "--speed" => {
                let Some(v) = args.next() else {
                    eprintln!("--speed requires a value");
                    std::process::exit(2);
                };
                speed = v.parse::<f32>().unwrap_or(1.0);
            }
            "--margin" => {
                let Some(v) = args.next() else {
                    eprintln!("--margin requires a value");
                    std::process::exit(2);
                };
                fit_margin = v.parse::<f32>().unwrap_or(0.9);
            }
            "--no-pma" => {
                prefer_pma_atlas = false;
            }
            _ => {
                if animation.is_none() {
                    animation = Some(arg);
                } else {
                    eprintln!("Unexpected argument: {arg:?}");
                    std::process::exit(2);
                }
            }
        }
    }
    let example = sanitize_example_arg(&raw_example);
    if example != raw_example {
        eprintln!("Example argument normalized: {raw_example:?} -> {example:?}");
    }
    let (json_path, atlas_path) = match pick_export_files(&example, prefer_pma_atlas) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(
        example,
        animation,
        speed,
        fit_margin,
        prefer_pma_atlas,
        json_path,
        atlas_path,
    );
    event_loop.run_app(&mut app).unwrap();
}
