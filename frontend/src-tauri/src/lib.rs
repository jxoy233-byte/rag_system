use tauri::{
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    Emitter, LogicalPosition, LogicalSize, Manager, PhysicalPosition,
};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

/// 暴露给前端的命令：在 header 空白处按下鼠标时调用，开始原生拖拽。
#[tauri::command]
fn start_window_drag(window: tauri::Window) -> Result<(), String> {
    window.start_dragging().map_err(|e| e.to_string())
}

/// 收起态：把窗口缩成 80x80（包含 padding 的小圆球尺寸），浮球贴在窗口右下角。
/// 展开态：把窗口还原成默认的 540x620 面板尺寸。
/// 两种切换都**锚定窗口右下角**不动：浮球永远在屏幕同一位置，窗口（面板）
/// 只向左上角方向铺开 / 收回，这样展开/折叠时浮球不会跳动。
/// mode: "ball" | "panel"
#[tauri::command]
async fn set_window_mode(window: tauri::Window, mode: String) -> Result<(), String> {
    let new_size = match mode.as_str() {
        "ball" => LogicalSize::new(80.0, 80.0),
        _ => LogicalSize::new(540.0, 620.0),
    };
    // 锚定右下角：记下当前物理尺寸 + 物理位置，算出新左上角
    let scale = window.scale_factor().map_err(|e| e.to_string())?;
    let cur_size = window.inner_size().map_err(|e| e.to_string())?;
    let cur_pos = window.outer_position().map_err(|e| e.to_string())?;
    let new_w_phys = (new_size.width * scale).round() as i32;
    let new_h_phys = (new_size.height * scale).round() as i32;
    let new_x = cur_pos.x + (cur_size.width as i32 - new_w_phys);
    let new_y = cur_pos.y + (cur_size.height as i32 - new_h_phys);
    window.set_size(new_size).map_err(|e| e.to_string())?;
    window
        .set_position(PhysicalPosition::new(new_x, new_y))
        .map_err(|e| e.to_string())?;
    // 收起态不允许调整大小，避免误拉伸
    let resizable = mode != "ball";
    window.set_resizable(resizable).map_err(|e| e.to_string())?;
    Ok(())
}

/// 退出应用（托盘菜单里用）。
#[tauri::command]
fn quit_app(app: tauri::AppHandle) {
    app.exit(0);
}

fn toggle_main_window(app: &tauri::AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let visible = w.is_visible().unwrap_or(false);
        if visible {
            let _ = w.hide();
        } else {
            let _ = w.show();
            let _ = w.set_focus();
            // 通知前端展开面板（因为 webview 隐藏时 JS 不跑，
            // 这里用事件让前端在窗口出现时把 expanded 置 true）
            let _ = w.emit("rag://show", ());
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .invoke_handler(tauri::generate_handler![start_window_drag, set_window_mode, quit_app])
        .setup(|app| {
            // 注：之前这里调了 setMovableByWindowBackground(true) 让整窗都可拖。
            // 用户反馈「拖拽范围太大」，现在改回只在 header 区域可拖，
            // 通过前端 data-tauri-drag-region + CSS -webkit-app-region 实现。

            // ---- 初始位置：屏幕右下角 ----
            // tauri.conf.json 里 center=false，没有 x/y；这里按主显示器尺寸算出右下角位置
            // （距右 40、距下 80 + 浮球高 80）。
            if let Some(window) = app.get_webview_window("main") {
                if let Ok(Some(monitor)) = window.primary_monitor() {
                    let size = monitor.size();
                    let scale = monitor.scale_factor();
                    let logical_w = size.width as f64 / scale;
                    let logical_h = size.height as f64 / scale;
                    let x = (logical_w - 120.0).max(0.0);
                    let y = (logical_h - 160.0).max(0.0);
                    let _ = window.set_position(LogicalPosition::new(x, y));
                }
                // ---- 失焦自动折叠 ----
                // 用户点击窗口外部（桌面、其他 App、托盘菜单外）时，窗口失焦。
                // 把信号转发给前端，让前端收起成浮球。
                let blur_handle = window.clone();
                window.on_window_event(move |event| {
                    if let tauri::WindowEvent::Focused(false) = event {
                        let _ = blur_handle.emit("rag://blur", ());
                    }
                });
            }

            // ---- 全局快捷键 ⌘⇧Space ----
            // 必须放在 Rust 侧：前端 JS 在 webview 隐藏时不会被执行，
            // 所以 JS 注册的快捷键在窗口最小化后会失效。
            let shortcut = Shortcut::new(
                Some(Modifiers::SUPER | Modifiers::SHIFT),
                Code::Space,
            );
            app.global_shortcut()
                .on_shortcut(shortcut, |app, _sc, event| {
                    if event.state() == ShortcutState::Pressed {
                        toggle_main_window(app);
                    }
                })
                .expect("failed to register global shortcut");

            // ---- 托盘菜单 ----
            let show_item = MenuItem::with_id(app, "show", "显示窗口", true, None::<&str>)?;
            let hide_item = MenuItem::with_id(app, "hide", "隐藏窗口", true, None::<&str>)?;
            let quit_item = MenuItem::with_id(app, "quit", "退出 RAG 助手", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&show_item, &hide_item, &quit_item])?;

            let _tray = TrayIconBuilder::with_id("main-tray")
                .icon(app.default_window_icon().unwrap().clone())
                .icon_as_template(true)
                .tooltip("RAG 助手")
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "show" => {
                        if let Some(w) = app.get_webview_window("main") {
                            let _ = w.show();
                            let _ = w.set_focus();
                            let _ = w.emit("rag://show", ());
                        }
                    }
                    "hide" => {
                        if let Some(w) = app.get_webview_window("main") {
                            let _ = w.hide();
                        }
                    }
                    "quit" => app.exit(0),
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        toggle_main_window(tray.app_handle());
                    }
                })
                .build(app)?;

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
