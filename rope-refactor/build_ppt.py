#!/usr/bin/env python3
"""Generate RoPE refactoring PPT with rendered PlantUML diagrams."""

import re, zlib, io
from pathlib import Path
import requests
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

BASE = Path('/Users/hellozkr/repos/LLM-Infer-Notes/rope-refactor')
BLUE = RGBColor(0x00, 0x33, 0x66)
LIGHT_BLUE = RGBColor(0x44, 0x72, 0xC4)
DARK = RGBColor(0x1A, 0x1A, 0x1A)
GRAY = RGBColor(0x66, 0x66, 0x66)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BG_LIGHT = RGBColor(0xF5, 0xF7, 0xFA)

# ---------- PlantUML render ----------

def encode6bit(b):
    if b < 10: return chr(48 + b)
    b -= 10
    if b < 26: return chr(65 + b)
    b -= 26
    if b < 26: return chr(97 + b)
    b -= 26
    if b == 0: return '-'
    if b == 1: return '_'
    return '?'

def encode3bytes(b1, b2, b3):
    c1 = b1 >> 2
    c2 = ((b1 & 0x3) << 4) | (b2 >> 4)
    c3 = ((b2 & 0xF) << 2) | (b3 >> 6)
    c4 = b3 & 0x3F
    return encode6bit(c1) + encode6bit(c2) + encode6bit(c3) + encode6bit(c4)

def encode64(data):
    result = []
    for i in range(0, len(data), 3):
        if i + 2 == len(data):
            result.append(encode3bytes(data[i], data[i+1], 0)[:-1])
        elif i + 1 == len(data):
            result.append(encode3bytes(data[i], 0, 0)[:-2])
        else:
            result.append(encode3bytes(data[i], data[i+1], data[i+2]))
    return ''.join(result)

def plantuml_encode(text):
    compressed = zlib.compress(text.encode('utf-8'))
    return encode64(compressed[2:-4])

def render_puml(text, path):
    encoded = plantuml_encode(text)
    url = f"http://www.plantuml.com/plantuml/png/{encoded}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(path, 'wb') as f:
        f.write(resp.content)
    print(f"  Rendered: {path.name} ({len(resp.content)} bytes)")

def extract_diagrams(md_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    pattern = r'```plantuml\n(.*?)```'
    matches = re.findall(pattern, content, re.DOTALL)
    titles = []
    for m in matches:
        start = content.find('```plantuml\n' + m)
        preceding = content[:start]
        heading_match = re.findall(r'^#+ (.+)$', preceding, re.MULTILINE)
        titles.append(heading_match[-1] if heading_match else 'Diagram')
    return list(zip(titles, matches))

# ---------- PPT helpers ----------

def add_slide_number(slide, num):
    left = Inches(9.5)
    top = Inches(6.9)
    txBox = slide.shapes.add_textbox(left, top, Inches(0.5), Inches(0.3))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = str(num)
    p.font.size = Pt(9)
    p.font.color.rgb = GRAY
    p.alignment = PP_ALIGN.RIGHT

def add_title_bar(slide, title_text, subtitle_text=None):
    """Add a blue title bar at top."""
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(10), Inches(1.1))
    shape.fill.solid()
    shape.fill.fore_color.rgb = BLUE
    shape.line.fill.background()

    txBox = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(9), Inches(0.6))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(26)
    p.font.color.rgb = WHITE
    p.font.bold = True

    if subtitle_text:
        txBox2 = slide.shapes.add_textbox(Inches(0.5), Inches(0.7), Inches(9), Inches(0.35))
        tf2 = txBox2.text_frame
        p2 = tf2.paragraphs[0]
        p2.text = subtitle_text
        p2.font.size = Pt(13)
        p2.font.color.rgb = RGBColor(0xBB, 0xCC, 0xDD)

def add_body_text(slide, text, left=0.6, top=1.4, width=8.8, height=5.0, size=14):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, line in enumerate(text.split('\n')):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(size)
        p.font.color.rgb = DARK
        p.space_after = Pt(4)
    return tf

def add_bullet_slide(slide, title, bullets, top=1.4):
    """Standard bullet-point slide with title bar."""
    add_title_bar(slide, title)
    txBox = slide.shapes.add_textbox(Inches(0.6), Inches(top), Inches(8.8), Inches(5.2))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, bullet in enumerate(bullets):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = bullet
        p.font.size = Pt(15)
        p.font.color.rgb = DARK
        p.space_after = Pt(8)
        p.level = bullet.count('  ') // 2  # simple indent hack
    return tf

def add_image_slide(slide, title, img_path, left=0.5, top=1.4, width=9.0):
    """Slide with title bar and a centered image."""
    add_title_bar(slide, title)
    if img_path and Path(img_path).exists():
        slide.shapes.add_picture(str(img_path), Inches(left), Inches(top), Inches(width))
    else:
        add_body_text(slide, f"[Image not available: {img_path}]", top=top)

def add_two_col(slide, title, left_content, right_content, top=1.4):
    """Two-column layout."""
    add_title_bar(slide, title)
    # Left
    txBox = slide.shapes.add_textbox(Inches(0.4), Inches(top), Inches(4.4), Inches(5.2))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, line in enumerate(left_content.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(13)
        p.font.color.rgb = DARK
        p.space_after = Pt(4)
    # Right
    txBox2 = slide.shapes.add_textbox(Inches(5.2), Inches(top), Inches(4.4), Inches(5.2))
    tf2 = txBox2.text_frame
    tf2.word_wrap = True
    for i, line in enumerate(right_content.split('\n')):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        p.text = line
        p.font.size = Pt(13)
        p.font.color.rgb = DARK
        p.space_after = Pt(4)

def add_code_block(slide, code, left=0.3, top=1.3, width=9.4, height=5.3, size=8):
    """Add a code block with dark background."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0x1E, 0x1E, 0x2E)
    shape.line.fill.background()

    txBox = slide.shapes.add_textbox(Inches(left + 0.15), Inches(top + 0.1), Inches(width - 0.3), Inches(height - 0.2))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, line in enumerate(code.strip().split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.name = 'Courier New'
        p.font.size = Pt(size)
        p.font.color.rgb = RGBColor(0xCD, 0xD6, 0xF4)
        p.space_after = Pt(1)

def add_table_slide(slide, title, headers, rows, top=1.4):
    """Slide with a table."""
    add_title_bar(slide, title)
    n_rows = len(rows) + 1
    n_cols = len(headers)
    table_shape = slide.shapes.add_table(n_rows, n_cols, Inches(0.5), Inches(top), Inches(9), Inches(0.4 * n_rows))
    table = table_shape.table

    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = BLUE
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(11)
            p.font.color.rgb = WHITE
            p.font.bold = True
            p.alignment = PP_ALIGN.CENTER

    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.text = val
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(10)
                p.font.color.rgb = DARK
            if i % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0xEE, 0xF1, 0xF5)

# ======================
# MAIN
# ======================
def main():
    # -- Step 1: Render PlantUML diagrams --
    print("Step 1: Rendering PlantUML diagrams...")
    diagrams = extract_diagrams(BASE / 'rope_design.md')
    img_paths = []
    for i, (title, text) in enumerate(diagrams):
        # Shorten titles for filenames
        short = title.replace(' ', '_').replace('/', '_')[:40]
        path = BASE / f'ppt_diagram_{i+1}.png'
        render_puml(text, path)
        img_paths.append((title, path))
    print(f"  {len(img_paths)} diagrams rendered.\n")

    # -- Step 2: Build PPT --
    print("Step 2: Building PPT...")
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)

    # ---- Slide 0: Cover ----
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(10), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = BLUE
    bg.line.fill.background()

    # Decorative line
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1), Inches(3.0), Inches(8), Inches(0.04))
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(0x44, 0x72, 0xC4)
    line.line.fill.background()

    txBox = slide.shapes.add_textbox(Inches(1), Inches(1.5), Inches(8), Inches(1.2))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = "RoPE模块注册机制重构"
    p.font.size = Pt(38)
    p.font.color.rgb = WHITE
    p.font.bold = True

    txBox2 = slide.shapes.add_textbox(Inches(1), Inches(3.3), Inches(8), Inches(0.8))
    tf2 = txBox2.text_frame
    p2 = tf2.paragraphs[0]
    p2.text = "MindIE-LLM Clean Code 优秀代码推荐  |  PR #285"
    p2.font.size = Pt(16)
    p2.font.color.rgb = RGBColor(0xBB, 0xCC, 0xDD)

    txBox3 = slide.shapes.add_textbox(Inches(1), Inches(5.5), Inches(8), Inches(0.5))
    tf3 = txBox3.text_frame
    p3 = tf3.paragraphs[0]
    p3.text = "zhaokerui  |  MindIE-LLM Team"
    p3.font.size = Pt(13)
    p3.font.color.rgb = RGBColor(0x99, 0xAA, 0xBB)

    add_slide_number(slide, 1)

    # ---- Slide 1: 为什么重构？----
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, "为什么重构？—— DeepSeek V4 带来的新挑战")

    bullets = [
        "●  原有设计：整个模型所有层共享同一 RoPE 实例",
        "      每层使用相同的 cos / sin table",
        "      初始化时创建一次，全局复用",
        "",
        "●  DeepSeek V4 新特性：不同层使用不同 Attention 机制",
        "      稠密层 → 标准 Attention",
        "      MoE 层 → SFA (Sparse Flash Attention)",
        "      推理层 → MLA (Multi-head Latent Attention)",
        "",
        "●  核心矛盾",
        "      不同 Attention 需要不同的 cos/sin → 必须每层独立创建 RoPE",
        "      相同配置的层应共享实例 → 需要缓存机制，避免重复计算",
        "      原有 if-elif 分支结构无法灵活适配不断增长的新模型",
    ]
    add_bullet_slide(slide, "为什么重构？—— DeepSeek V4 带来的新挑战", bullets, top=1.2)

    add_slide_number(slide, 2)

    # ---- Slide 2: 重构前根本缺陷：违反开闭原则 ----
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, "重构前的根本缺陷：违反开闭原则")

    left_content = """重构前：所有模型共享一个 position_embedding 脚本

◆ 全部模型使用同一个文件
    position_rotary_embedding.py
    包含所有模型的 RoPE 创建逻辑

◆ 新模型接入 → 必须修改这个文件
    if rope_type == "default": ...
    elif rope_type == "yarn": ...
    elif rope_type == "new_model": ...   ← 每次加分支

◆ 违反开闭原则 (OCP)
    对修改开放：每个新模型都要改公共脚本
    对扩展封闭：没有提供模型自定义的扩展点

◆ 风险
    修改公共脚本可能影响已有模型
    多人并行开发时易冲突
    无法独立测试单个模型的 RoPE"""
    right_content = """重构后：每个模型自己实现 RoPE + 按需复用框架缓存

◆ 模型独立实现
    每个模型在自身目录定义 RoPE 子类
    deepseek_v3_yarn_scaling_rope.py
    minmax_rope.py、qwen2_rope.py ... （各自文件）

◆ 通过注册复用框架能力
    @register_rope_type("deepseek_yarn")
    @cached_rope_factory
    一行注册即可获得框架的 cos/sin 缓存能力

◆ 符合开闭原则 (OCP)
    对扩展开放：新模型加新文件即可
    对修改封闭：框架代码零修改

◆ 自由度
    不想用框架缓存？不注册即可
    想自定义缓存逻辑？自己实现
    注册 = 自愿加入框架缓存机制"""

    add_two_col(slide, "重构前的根本缺陷：违反开闭原则", left_content, right_content, top=1.2)
    add_slide_number(slide, 3)

    # ---- Slide 3: 重构前 vs 重构后 ----
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    left_content = """重构前的问题

◆ RopeScaling 配置耦合
    rope_type 写死在 validator 枚举中，
    新增类型需修改框架代码

◆ 工厂逻辑集中硬编码
    position_rotary_embedding.py 中
    if-elif 分支判断 rope_type，
    扩展性差

◆ cos/sin 耦合在 AttentionMetadata
    attention 后端直接依赖 RoPE 内部
    数据结构，关注点不分离

◆ 无缓存，无法按需共享
    要么全层共享一个实例，
    要么每层重复创建"""
    right_content = """重构后的优势

◆ BaseRopeScaling + 模型特化子类
    每个模型定义自己的配置类，
    框架代码零修改

◆ 注册机制 + 工厂模式
    @register_rope_type 装饰器一行注册，
    自动发现，无硬编码分支

◆ cos/sin 解耦
    通过 RoPE 实例 buffer 管理，
    作为独立参数传入 attention 后端

◆ 自动缓存
    cached_rope_factory 按参数自动缓存，
    相同配置共享，不同配置隔离"""

    add_two_col(slide, "重构前 vs 重构后", left_content, right_content, top=1.2)
    add_slide_number(slide, 4)

    # ---- Slide 5: RoPE 类图 ----
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_image_slide(slide, "RoPE 类继承层次", img_paths[0][1])
    add_slide_number(slide, 5)

    # ---- Slide 6: get_rope 流程图 ----
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_image_slide(slide, "get_rope — Factory + Registry + Cache", img_paths[1][1])
    add_slide_number(slide, 6)

    # ---- Slide 7: 运行时序图 ----
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_image_slide(slide, "运行时序图 — DeepSeek V3.2 推理流程", img_paths[2][1])
    add_slide_number(slide, 7)

    # ---- Slide 8: 设计模式总结 ----
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_table_slide(slide, "设计模式组合运用",
        ['设计模式', '应用位置', '解决的问题'],
        [
            ['注册表模式 (Registry)', 'registry.py · _ROPE_REGISTRY', '类型标识→工厂函数映射，支持运行时动态注册'],
            ['工厂方法 (Factory Method)', '__init__.py · get_rope()', '根据 rope_type 动态选择实现，调用者无需知道类名'],
            ['模板方法 (Template)', 'base.py · _compute_cos_sin_cache()', '基类定义算法骨架，子类重写特定步骤'],
            ['装饰器 (Decorator)', 'registry.py · @cached_rope_factory', '透明地为 factory 函数增加缓存能力'],
            ['外观模式 (Facade)', '__init__.py · get_rope()', '统一入口，封装 registry 查询+factory 调度+参数处理'],
        ],
        top=1.2)
    add_slide_number(slide, 8)

    # ---- Slide 9: 核心代码 - Registry ----
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, "核心实现：Registry 注册 + 缓存装饰器")
    code = """def register_rope_type(rope_type: str) -> Callable:
    def decorator(factory_func):
        if rope_type in _ROPE_REGISTRY:
            raise ValueError(f"'{rope_type}' already registered")
        _ROPE_REGISTRY[rope_type] = factory_func
        return factory_func
    return decorator

@register_rope_type("deepseek_yarn")      # ← 一行注册
@cached_rope_factory                      # ← 一行加缓存
def _create_deepseek_scaling_rope(head_size, rotary_dim, ..., rope_config):
    return DeepseekV3YarnRotaryEmbedding(rotary_dim, ...)

# 新增模型：仅需 ①实现子类  ②写factory函数+两个装饰器
# 零处已有代码修改"""
    add_code_block(slide, code, left=0.4, top=1.2, width=9.2, height=5.5, size=9)
    add_slide_number(slide, 9)

    # ---- Slide 10: 核心代码 - cos/sin 解耦 ----
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, "核心实现：cos/sin 与 Attention 后端的解耦")

    left_code = """# Before — cos/sin 耦合在 Metadata
class SfaMetadata(AttentionMetadata):
    cos_table: Tensor
    sin_table: Tensor

    @staticmethod
    def from_model_input(model_inputs,
            cos_table, sin_table, mask, ...):
        return SfaMetadata(
            cos_table=cos_table,
            sin_table=sin_table, ...)
# Attention backend 直接依赖 RoPE 内部结构
q_pe = npu_interleave_rope(
    q_pe,
    attn_metadata.cos_table,
    attn_metadata.sin_table)"""

    right_code = """# After — cos/sin 作为独立参数
class SfaMetadata(AttentionMetadata):
    # cos/sin 已移除，回归纯元信息角色
    seq_lens, slot_mapping, block_tables, ...

class SfaBackendImpl:
    def decode_qk_rope(self, hidden_state, ...,
            attn_metadata, cos, sin):  # ← 独立参数
        q_pe = npu_interleave_rope(q_pe, cos, sin)

# RoPE 实例自身管理 cos/sin buffer
rope_emb.set_cos_sin_indexed_cache(positions)
# attention 后端不再关心 cos/sin 来源"""

    # Left code
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.15), Inches(1.3), Inches(4.7), Inches(5.5))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0x1E, 0x1E, 0x2E)
    shape.line.fill.background()
    txBox = slide.shapes.add_textbox(Inches(0.25), Inches(1.35), Inches(4.5), Inches(5.4))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, line in enumerate(left_code.strip().split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.name = 'Courier New'
        p.font.size = Pt(7.5)
        p.font.color.rgb = RGBColor(0xCD, 0xD6, 0xF4)
        p.space_after = Pt(1)

    # Right code
    shape2 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(5.1), Inches(1.3), Inches(4.7), Inches(5.5))
    shape2.fill.solid()
    shape2.fill.fore_color.rgb = RGBColor(0x1E, 0x2E, 0x1E)
    shape2.line.fill.background()
    txBox2 = slide.shapes.add_textbox(Inches(5.2), Inches(1.35), Inches(4.5), Inches(5.4))
    tf2 = txBox2.text_frame
    tf2.word_wrap = True
    for i, line in enumerate(right_code.strip().split('\n')):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        p.text = line
        p.font.name = 'Courier New'
        p.font.size = Pt(7.5)
        p.font.color.rgb = RGBColor(0xA6, 0xE2, 0xA0)
        p.space_after = Pt(1)

    # Label
    label = slide.shapes.add_textbox(Inches(0.15), Inches(1.1), Inches(4.7), Inches(0.25))
    p = label.text_frame.paragraphs[0]
    p.text = "✗ 重构前"
    p.font.size = Pt(12)
    p.font.color.rgb = RGBColor(0xFF, 0x66, 0x66)
    p.font.bold = True

    label2 = slide.shapes.add_textbox(Inches(5.1), Inches(1.1), Inches(4.7), Inches(0.25))
    p2 = label2.text_frame.paragraphs[0]
    p2.text = "✓ 重构后"
    p2.font.size = Pt(12)
    p2.font.color.rgb = RGBColor(0x66, 0xFF, 0x66)
    p2.font.bold = True

    add_slide_number(slide, 10)

    # ---- Slide 11: 结果 —— 支撑新模型 ----
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, "结果一：支撑 MindIE-LLM 多模型 RoPE 计算")

    bullets = [
        "●  已适配模型",
        "      DeepSeek V3 — deepseek_yarn 类型，mscale + mscale_all_dim 双因子缩放",
        "      Qwen2 — yarn 类型，标准 YaRN 上下文扩展",
        "      Qwen3 — default 类型，标准 RoPE",
        "",
        "●  缓存效率",
        "      DeepSeek V3: 61 层 → 仅创建 1 个 RoPE 实例（所有层配置相同）",
        "      DeepSeek V4: N 种 Attention → 仅创建 M 个实例（M = 不同配置数）",
        '      与原有"全局共享一个 RoPE"的实现效率完全一致',
        "",
        "●  新增模型零侵入",
        "      实现子类 → 注册 factory → 完成",
        "      不需要修改 registry.py、__init__.py、attention backend 任何已有代码",
    ]
    add_bullet_slide(slide, "结果一：支撑 MindIE-LLM 多模型 RoPE 计算", bullets, top=1.2)
    add_slide_number(slide, 11)

    # ---- Slide 12: MiniMax 案例 ----
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, "结果二：MiniMax 模型适配 — AI 时代的高效实践")

    left = """适配案例：MiniMax-M1

◆ 背景
    MiniMax-M1 使用自定义 RoPE 实现，
    需要从原生 transformers 代码迁移
    到 MindIE-LLM 框架

◆ 流程
    ① 从 transformers 源码提取 RoPE 逻辑
    ② 继承 YarnScalingRotaryEmbedding 基类
    ③ 重写 _compute_cos_sin_cache()
    ④ 编写 factory 函数 + 双装饰器注册
    ⑤ 在 config 中定义 MiniMaxRopeScaling

◆ 结果
    AI 辅助开发，总耗时仅 1 小时
    仅遇到 1 个运行时错误（dtype 不匹配）
    修复后即通过全部测试"""

    right = """为什么能这么快？

◆ 清晰的扩展点
    基类已定义好模板方法，
    只需重写差异逻辑，不需要理解
    整个框架的 RoPE 调用链路

◆ 声明式注册
    装饰器即文档，AI 能直接从
    已有注册代码中学习模式并生成

◆ 独立的测试边界
    可以单独测试自定义 RoPE，
    不需要启动完整推理服务

◆ AI 时代的代码设计哲学

    好的设计 = AI 能理解的设计
    清晰的接口 + 一致的命名
    + 明确的扩展点
    → AI 能快速生成可靠代码

    本设计的注册机制正是例证：
    MiniMax 适配仅 1 小时即完成"""
    add_two_col(slide, "结果二：MiniMax 模型适配 — AI 时代的高效实践", left, right, top=1.2)
    add_slide_number(slide, 12)

    # ---- Slide 13: AI 时代的设计 ----
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, "AI 时代的代码设计：好设计让 AI 写出好代码")

    bullets = [
        "●  我们的设计实践验证了一个观点：",
        "      在 AI 辅助编程时代，好的代码设计能让 AI 更快速地理解、学习并生成正确代码",
        "",
        "●  为什么本设计对 AI 友好？",
        "      ① 模块职责单一：registry 只管注册，base 只管计算，factory 只管创建",
        "          AI 不需要理解全局才能写对局部代码",
        "      ② 装饰器声明式：@register_rope_type + @cached_rope_factory",
        "          意图直接表达，AI 从上下文可以学习并复现这个模式",
        "      ③ 继承层次清晰：Base → YaRN → DeepSeekV3YaRN",
        "          每一层只增加一个变化维度，AI 知道该覆盖哪个方法",
        "      ④ 命名符合社区惯例：register / factory / cache / get_xxx",
        "          AI 训练数据中大量出现这些模式，理解成本极低",
        "",
        "●  MiniMax 1 小时适配 = 这个设计哲学的实证",
        "      从 transformers 原生代码 → 框架集成，仅遇到 1 个运行时错误",
    ]
    add_bullet_slide(slide, "AI 时代的代码设计：好设计让 AI 写出好代码", bullets, top=1.2)
    add_slide_number(slide, 13)

    # ---- Slide 14: 总结 ----
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(10), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = BLUE
    bg.line.fill.background()

    txBox = slide.shapes.add_textbox(Inches(1), Inches(0.8), Inches(8), Inches(1))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = "总结"
    p.font.size = Pt(32)
    p.font.color.rgb = WHITE
    p.font.bold = True

    summary_items = [
        "机制性框架设计 — 注册 + 缓存 + 工厂，对扩展开放、对修改关闭",
        "cos/sin 解耦 — Attention 后端回归纯粹的元信息角色",
        "多模型支撑 — DeepSeek V3/V4、Qwen2/3、MiniMax 均已适配",
        "AI 友好 — MiniMax 适配仅 1 小时，1 个运行时错误",
        "可推广 — 注册机制可复用到 Attention Backend、MoE Router 等模块",
    ]

    txBox2 = slide.shapes.add_textbox(Inches(1), Inches(2.2), Inches(8), Inches(4))
    tf2 = txBox2.text_frame
    for i, item in enumerate(summary_items):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        p.text = f"  {item}"
        p.font.size = Pt(18)
        p.font.color.rgb = RGBColor(0xCC, 0xDD, 0xEE)
        p.space_after = Pt(18)

    txBox3 = slide.shapes.add_textbox(Inches(1), Inches(6.2), Inches(8), Inches(0.5))
    tf3 = txBox3.text_frame
    p3 = tf3.paragraphs[0]
    p3.text = "谢谢  |  zhaokerui"
    p3.font.size = Pt(16)
    p3.font.color.rgb = RGBColor(0x99, 0xAA, 0xBB)

    add_slide_number(slide, 14)

    # -- Save --
    out = BASE / 'RoPE模块注册机制重构_好代码推荐.pptx'
    prs.save(out)
    print(f"\nPPT saved: {out}")
    print(f"  Slides: {len(prs.slides)}")
    print(f"  Diagrams retained in: {[p.name for _, p in img_paths]}")


if __name__ == '__main__':
    main()
