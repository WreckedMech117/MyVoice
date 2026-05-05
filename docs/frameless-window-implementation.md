# Frameless Window Implementation - Task Breakdown

**Project:** MyVoice TTS Desktop Application
**Feature:** Custom Frameless Window with Sleek Design
**Estimated Total Effort:** 7-10 hours

---

## Task 1: Create Custom Title Bar Component

**File:** `src/myvoice/ui/components/custom_title_bar.py`

**Requirements:**
- Create `CustomTitleBar` class inheriting from `QWidget`
- Implement compact design (24-28px height)
- Add window control buttons: minimize, maximize/restore, close
- Add draggable title area with app name/icon
- Implement mouse event handlers for window dragging
- Handle maximize/restore state changes
- Support theme styling via QSS

**Implementation Details:**
```python
class CustomTitleBar(QWidget):
    - __init__(parent): Initialize title bar with controls
    - _create_ui(): Build title and button layout
    - _setup_buttons(): Create min/max/close buttons
    - mousePressEvent(): Capture drag start position
    - mouseMoveEvent(): Handle window dragging
    - mouseDoubleClickEvent(): Toggle maximize/restore
    - _on_minimize_clicked(): Minimize window
    - _on_maximize_clicked(): Toggle maximize/restore
    - _on_close_clicked(): Close window
```

**Acceptance Criteria:**
- [ ] Title bar renders with proper height and layout
- [ ] Window can be dragged by clicking title bar
- [ ] Minimize button works correctly
- [ ] Maximize/restore button toggles window state
- [ ] Close button closes application
- [ ] Double-click on title bar maximizes/restores
- [ ] Theme styling applies correctly

**Estimated Time:** 2-3 hours

---

## Task 2: Implement Window Resize Grips

**Files:**
- `src/myvoice/ui/components/resize_grip.py` (new)
- Modifications to `main_window.py`

**Requirements:**
- Create `SideGrip` class for edge resizing (left, top, right, bottom)
- Implement corner grips using `QSizeGrip`
- Set appropriate cursors for each grip direction
- Position grips dynamically during window resize
- Ensure grips work with frameless window flags

**Implementation Details:**

### SideGrip Class:
```python
class SideGrip(QWidget):
    def __init__(self, parent, edge):
        # edge: 'left', 'top', 'right', 'bottom'
    - Grip size: 4-6px for sleek design
    - Set appropriate cursor based on edge
    - Handle mouse events for resizing
```

### MainWindow Integration:
```python
# In MainWindow.__init__():
- Create 4 corner grips (QSizeGrip)
- Create 4 edge grips (SideGrip)
- Set grip sizes and properties

# Override resizeEvent():
- Position corner grips at window corners
- Position edge grips along window edges
- Ensure proper z-ordering
```

**Acceptance Criteria:**
- [ ] All 4 corners have visible/functional resize grips
- [ ] All 4 edges have functional resize areas
- [ ] Appropriate cursors appear on hover
- [ ] Window resizes smoothly from all directions
- [ ] Grips reposition correctly during resize
- [ ] Grips maintain proper size constraints (min/max)

**Estimated Time:** 2-3 hours

---

## Task 3: Integrate Frameless Window into MainWindow

**File:** `src/myvoice/ui/main_window.py`

**Requirements:**
- Modify window flags to use `FramelessWindowHint`
- Integrate custom title bar as first layout element
- Initialize and position resize grips
- Update window behavior for frameless mode
- Preserve existing always-on-top functionality
- Maintain window geometry and state

**Implementation Details:**

### Window Setup Changes:
```python
def _setup_window_properties(self):
    # Line 102-106 replacement
    self.setWindowFlags(
        Qt.WindowType.FramelessWindowHint |
        Qt.WindowType.Window
    )

    # Optional: Add translucent background for rounded corners
    # self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
```

### UI Integration:
```python
def _create_ui(self):
    # Add custom title bar as first element
    self.title_bar = CustomTitleBar(self)
    main_layout.addWidget(self.title_bar)

    # Initialize resize grips
    self._create_resize_grips()

    # Rest of existing UI...

def _create_resize_grips(self):
    # Create corner grips
    self.corner_grips = []
    for i in range(4):
        grip = QSizeGrip(self)
        self.corner_grips.append(grip)

    # Create edge grips
    self.edge_grips = {
        'left': SideGrip(self, 'left'),
        'top': SideGrip(self, 'top'),
        'right': SideGrip(self, 'right'),
        'bottom': SideGrip(self, 'bottom')
    }

def resizeEvent(self, event):
    super().resizeEvent(event)
    self._position_grips()

def _position_grips(self):
    rect = self.rect()
    grip_size = 6

    # Position corner grips
    self.corner_grips[0].move(0, 0)  # Top-left
    self.corner_grips[1].move(rect.right() - grip_size, 0)  # Top-right
    self.corner_grips[2].move(rect.right() - grip_size, rect.bottom() - grip_size)  # Bottom-right
    self.corner_grips[3].move(0, rect.bottom() - grip_size)  # Bottom-left

    # Position edge grips
    # Left, top, right, bottom edges...
```

**Acceptance Criteria:**
- [ ] Window displays without native title bar
- [ ] Custom title bar appears at top of window
- [ ] All existing UI elements render correctly below title bar
- [ ] Window can be moved via custom title bar
- [ ] Window can be resized from all edges and corners
- [ ] Always-on-top functionality still works
- [ ] Window geometry persists correctly
- [ ] No visual glitches during resize/move

**Estimated Time:** 1-2 hours

---

## Task 4: Update Theme Stylesheets for Frameless Design

**Files:**
- `src/myvoice/ui/styles/minimal_dark.qss`
- `src/myvoice/ui/styles/minimal_light.qss`

**Requirements:**
- Add styling for custom title bar
- Style window control buttons (minimize, maximize, close)
- Add hover/pressed states for title bar buttons
- Style resize grips (optional visual indicators)
- Add rounded corners to main window (optional)
- Ensure consistent dark/light theme support

**Implementation Details:**

### Title Bar Styling:
```css
/* Custom Title Bar */
QWidget[objectName="custom_title_bar"] {
    background-color: #0f172a;
    border-bottom: 1px solid #334155;
    min-height: 28px;
    max-height: 28px;
}

/* Title Bar Label */
QLabel[objectName="title_label"] {
    color: #f8fafc;
    font-weight: 600;
    font-size: 9pt;
    padding-left: 8px;
}

/* Window Control Buttons */
QPushButton[objectName="minimize_button"],
QPushButton[objectName="maximize_button"],
QPushButton[objectName="close_button"] {
    background: transparent;
    border: none;
    border-radius: 4px;
    min-width: 32px;
    max-width: 32px;
    min-height: 24px;
    max-height: 24px;
    color: #cbd5e1;
    font-size: 14px;
}

QPushButton[objectName="minimize_button"]:hover,
QPushButton[objectName="maximize_button"]:hover {
    background-color: #1e293b;
}

QPushButton[objectName="close_button"]:hover {
    background-color: #dc2626;
    color: white;
}

/* Main Window with Rounded Corners (optional) */
QMainWindow {
    background-color: #020617;
    border: 1px solid #334155;
    border-radius: 8px;
}
```

**Acceptance Criteria:**
- [ ] Title bar has proper background color and height
- [ ] Window control buttons are styled correctly
- [ ] Hover states work on all buttons
- [ ] Close button has distinctive red hover state
- [ ] Dark theme applied successfully
- [ ] Light theme applied successfully
- [ ] Rounded corners render properly (if enabled)
- [ ] Border visible around window edge

**Estimated Time:** 1-2 hours

---

## Task 5: Testing and Polish

**Requirements:**
- Test all window operations (move, resize, minimize, maximize, close)
- Test theme switching with frameless design
- Verify always-on-top behavior
- Test on Windows 10 and Windows 11
- Fix any visual glitches or edge cases
- Optimize performance
- Update documentation

**Test Cases:**

### Functional Testing:
- [ ] Window dragging works smoothly
- [ ] All 8 resize directions work (4 corners, 4 edges)
- [ ] Minimize button minimizes to taskbar
- [ ] Maximize button toggles fullscreen correctly
- [ ] Close button closes application cleanly
- [ ] Double-click title bar maximizes/restores
- [ ] Always-on-top setting persists

### Theme Testing:
- [ ] Dark theme renders correctly on frameless window
- [ ] Light theme renders correctly on frameless window
- [ ] Theme switching works without visual glitches
- [ ] Title bar buttons are visible in both themes
- [ ] Window border is visible in both themes

### Edge Cases:
- [ ] Window doesn't go off-screen when dragging
- [ ] Resize respects min/max size constraints
- [ ] Window state persists after minimize/restore
- [ ] Multi-monitor support works correctly
- [ ] High DPI displays render correctly

### Performance:
- [ ] No lag during window dragging
- [ ] Smooth resizing from all directions
- [ ] No memory leaks from grip objects
- [ ] Theme application is instant

**Bug Fixes:**
- Document and fix any issues found during testing
- Ensure cross-version compatibility (Windows 10/11)

**Estimated Time:** 2 hours

---

## Implementation Order

Execute tasks in numerical order:
1. **Task 1** → Create title bar component (foundation)
2. **Task 2** → Implement resize grips (resizing functionality)
3. **Task 3** → Integrate into MainWindow (assembly)
4. **Task 4** → Apply theme styling (visual polish)
5. **Task 5** → Test and fix issues (quality assurance)

---

## Technical Notes

### Window Flags:
```python
Qt.WindowType.FramelessWindowHint  # Remove native title bar
Qt.WindowType.Window                # Keep as top-level window
# Preserve existing always-on-top flag handling
```

### Key Considerations:
- Maintain existing window size constraints (320x160 min, 600x280 max)
- Preserve all existing functionality (voice selector, text input, emotion slider, etc.)
- Title bar height reduces available content area by ~28px
- Grip sizes should be subtle (4-6px) for sleek appearance
- Z-ordering: Corner grips on top of edge grips
- Mouse event propagation must be handled correctly

### Dependencies:
- PyQt6.QtWidgets: QWidget, QSizeGrip, QHBoxLayout, QPushButton, QLabel
- PyQt6.QtCore: Qt, QPoint, QRect, QEvent
- PyQt6.QtGui: QIcon, QCursor

### Alternative Approach:
If implementation proves complex, consider evaluating `PyQt6-Frameless-Window` package:
```bash
pip install PyQt6-Frameless-Window
```

However, custom implementation provides:
- Full control over design and behavior
- No external dependencies
- Optimized for compact desktop design
- Integrated with existing theme system

---

## Success Criteria

**Feature is complete when:**
- ✅ Window has no native title bar
- ✅ Custom title bar is functional and styled
- ✅ Window can be moved by dragging title bar
- ✅ Window can be resized from all 8 directions
- ✅ Window controls (min/max/close) work correctly
- ✅ Both themes (dark/light) render properly
- ✅ Always-on-top functionality preserved
- ✅ All existing features work unchanged
- ✅ No visual glitches or performance issues
- ✅ Application maintains compact, sleek appearance

---

**Document Version:** 1.0
**Created:** 2025-09-29
**Status:** Planning