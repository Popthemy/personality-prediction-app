from pathlib import Path
from copy import deepcopy
import hashlib, json, re, zipfile
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent
SOURCE = Path(r'C:/Users/STADIUM B C/Downloads/rewritten_chapter_one.docx')
OUT = ROOT.parents[1] / 'output/documents/rewritten_chapter_three.docx'
OUT.parent.mkdir(parents=True, exist_ok=True)
FONT = 'C:/Windows/Fonts/times.ttf'
BOLD = 'C:/Windows/Fonts/timesbd.ttf'

def diagram(name, height):
    im = Image.new('RGB', (1600, height), 'white')
    return im, ImageDraw.Draw(im)

def box(draw, rect, text, size=35, bold=False, shape='box'):
    x1,y1,x2,y2=rect
    if shape=='round': draw.rounded_rectangle(rect, radius=28, fill='white', outline='black', width=3)
    elif shape=='diamond': draw.polygon([(int((x1+x2)/2),y1),(x2,int((y1+y2)/2)),(int((x1+x2)/2),y2),(x1,int((y1+y2)/2))],fill='white',outline='black',width=3)
    else: draw.rectangle(rect,fill='white',outline='black',width=3)
    font=ImageFont.truetype(BOLD if bold else FONT,size)
    bounds=draw.multiline_textbbox((0,0),text,font=font,spacing=8,align='center')
    tw,th=bounds[2]-bounds[0],bounds[3]-bounds[1]
    assert tw < x2-x1-12, (text,tw,x2-x1)
    draw.multiline_text(((x1+x2-tw)/2,(y1+y2-th)/2-bounds[1]),text,font=font,fill='black',spacing=8,align='center')

def arrow(draw, points, label=None, labelpos=None):
    draw.line(points,fill='black',width=3)
    x,y=points[-1]; px,py=points[-2]
    import math
    angle=math.atan2(y-py,x-px)
    p1=(x-16*math.cos(angle-.45),y-16*math.sin(angle-.45))
    p2=(x-16*math.cos(angle+.45),y-16*math.sin(angle+.45))
    draw.polygon([(x,y),p1,p2],fill='black')
    if label:
        draw.text(labelpos,label,font=ImageFont.truetype(FONT,29),fill='black')

im,d=diagram('architecture',1120)
box(d,(460,15,1140,125),'Researcher',bold=True)
box(d,(330,210,1270,340),'Django researcher interface\nSettings, run status and results')
box(d,(350,425,1250,565),'Experiment coordination\nData preparation and eight conditions')
box(d,(30,425,285,565),'PANDORA\ndata files',size=32)
box(d,(90,675,1510,835),'Machine learning services\nSelection → BERT → training-only GAN where enabled\nPooled Lasso or ElasticNet / sequence LSTM → evaluation',size=34)
box(d,(130,950,750,1095),'Research database\nRun and prediction records')
box(d,(850,950,1470,1095),'Saved artifacts\nModels, metrics and audits')
arrow(d,[(800,125),(800,210)])
arrow(d,[(800,340),(800,425)])
arrow(d,[(285,495),(350,495)])
arrow(d,[(800,565),(800,675)])
arrow(d,[(440,835),(440,950)])
arrow(d,[(1160,835),(1160,950)])
im.save(ROOT/'architecture.png')

im,d=diagram('preparation',1230)
for x,t in [(20,'Training file'),(550,'Validation file'),(1080,'Test file')]:
    box(d,(x,15,x+500,115),t,bold=True)
    arrow(d,[(x+250,115),(x+250,210)])
    box(d,(x,210,x+500,370),'Load author, comment\nand five trait scores',size=33)
    arrow(d,[(x+250,370),(x+250,455)])
    box(d,(x,455,x+500,615),'Clean and group by author\nRetain eligible authors',size=31)
    arrow(d,[(x+250,615),(x+250,705)])
box(d,(250,705,1350,850),'Verify source scales and distinct author membership\nRecord selected authors and split counts',size=34)
for x,t in [(20,'Training authors\nFit selector, GAN,\nscaler and models'),(550,'Validation authors\nSelect LSTM epoch\nand decision thresholds'),(1080,'Test authors\nFinal held-out\nperformance reporting')]:
    arrow(d,[(800,850),(800,905),(x+250,905),(x+250,980)])
    box(d,(x,980,x+500,1210),t,size=33)
im.save(ROOT/'preparation.png')

im,d=diagram('pipeline',1770)
box(d,(345,15,1255,130),'Prepared author-level train, validation and test data',size=34,bold=True)
box(d,(35,245,765,410),'Full-history baseline\nRetain every available cleaned comment',size=34)
box(d,(835,245,1565,410),'Q-learning selection\nTrain policy on training authors only',size=34)
arrow(d,[(800,130),(800,190),(400,190),(400,245)])
arrow(d,[(800,190),(1200,190),(1200,245)])
box(d,(345,520,1255,650),'BERT encoding for each selection setting\n768-dimensional vector per comment',size=34)
arrow(d,[(400,410),(400,465),(800,465),(800,520)])
arrow(d,[(1200,410),(1200,465),(800,465)])
box(d,(45,760,755,900),'Mean-pooled author vectors\nLasso or ElasticNet branch',size=36)
box(d,(845,760,1555,900),'Ordered comment vectors\nStacked bidirectional LSTM branch',size=34)
arrow(d,[(800,650),(800,710),(400,710),(400,760)])
arrow(d,[(800,710),(1200,710),(1200,760)])
for x in [45,845]:
    arrow(d,[(x+355,900),(x+355,980)])
    box(d,(x,980,x+710,1130),'Fit with real training data\nCompare no GAN with training-only GAN',size=32)
    arrow(d,[(x+355,1130),(x+355,1225)])
box(d,(230,1225,1370,1385),'Eight fitted conditions\nValidation: retain LSTM epoch and select trait thresholds\nTraining labels define true Low/High boundaries',size=33)
box(d,(330,1490,1270,1640),'Apply fixed conditions and thresholds to test authors\nRecord scores, metrics and matched comparisons',size=33)
arrow(d,[(800,1385),(800,1490)])
d.text((270,1690),'Validation and test observations are never GAN training examples.',font=ImageFont.truetype(FONT,33),fill='black')
im.save(ROOT/'pipeline.png')

im,d=diagram('lstm',1330)
items=[('Ordered BERT comment embeddings\nBatch × sequence length × 768',10,165),('Pad sequences and retain true lengths',245,350),('Two stacked bidirectional LSTM layers\n128 hidden units in each direction',430,585),('Mean pooling over valid positions\n256-dimensional author representation',665,820),('Layer normalization and dropout\n64-unit dense layer, ReLU and dropout',900,1055),('Five continuous outputs\nO       C       E       A       N',1135,1300)]
for i,(t,y1,y2) in enumerate(items):
    box(d,(300,y1,1300,y2),t,size=37)
    if i: arrow(d,[(800,items[i-1][2]),(800,y1)])
im.save(ROOT/'lstm.png')

im,d=diagram('erd',1430)
box(d,(590,10,1010,120),'Researcher\nuser id',size=32,bold=True)
box(d,(310,240,1290,420),'PANDORA_EXPERIMENT_RUN\nPK id     FK researcher id\nRun id, settings, status and artifact directory',size=32)
arrow(d,[(800,120),(800,240)],'1 : many',(815,170))
rects=[(15,570,515,790),(550,570,1050,790),(1085,570,1585,790)]
texts=['CONDITION_RESULT\nPK id     FK run id\nCondition and metrics','THRESHOLD_RESULT\nPK id     FK run id\nCondition, trait and threshold','DATASET_ALLOCATION\nPK id     FK run id\nAuthor and source split']
for rect,t in zip(rects,texts):
    box(d,rect,t,size=29)
    cx=(rect[0]+rect[2])//2
    arrow(d,[(800,420),(800,495),(cx,495),(cx,570)])
    d.text((cx+10,525),'1 : many',font=ImageFont.truetype(FONT,26),fill='black')
box(d,(390,955,1210,1155),'PANDORA_PREDICTION_RUN\nPK id     FK researcher id     FK experiment run id\nSelected condition and batch summary',size=30)
arrow(d,[(1290,330),(1595,330),(1595,870),(800,870),(800,955)],'0..1 experiment per prediction run',(830,892))
box(d,(390,1250,1210,1415),'PANDORA_TEST_PROFILE\nPK id     FK prediction run id\nAuthor, true scores and predicted scores',size=31)
arrow(d,[(800,1155),(800,1250)],'1 : many',(815,1188))
d.text((15,30),'PK = primary key\nFK = foreign key',font=ImageFont.truetype(FONT,28),fill='black')
d.text((15,970),'Each prediction run\nalso belongs to\none researcher.',font=ImageFont.truetype(FONT,28),fill='black')
im.save(ROOT/'erd.png')

doc=Document(SOURCE)
body_props=deepcopy(doc.paragraphs[3]._p.pPr)
head_props=deepcopy(doc.paragraphs[2]._p.pPr)
title_props=deepcopy(doc.paragraphs[0]._p.pPr)
for el in list(doc._element.body):
    if el.tag!=qn('w:sectPr'): doc._element.body.remove(el)

def set_font(run,bold=False,size=12):
    run.font.name='Times New Roman'; run.font.size=Pt(size)
    run.font.color.rgb=RGBColor(0,0,0); run.font.bold=bold

def paragraph(text,kind='body'):
    p=doc.add_paragraph()
    props=title_props if kind=='title' else head_props if kind=='heading' else body_props
    p._p.insert(0,deepcopy(props))
    if kind=='title':
        p.style=doc.styles['Title']
        p.paragraph_format.first_line_indent=Inches(0)
        p.paragraph_format.keep_with_next=True
        borders=OxmlElement('w:pBdr')
        for side in ['top','left','bottom','right','between','bar']:
            border=OxmlElement('w:'+side);border.set(qn('w:val'),'nil');borders.append(border)
        p._p.pPr.append(borders)
    if kind=='heading':
        p.paragraph_format.keep_with_next=True
        p.paragraph_format.first_line_indent=Inches(0)
        outline=OxmlElement('w:outlineLvl')
        outline.set(qn('w:val'),'1' if re.match(r'3\.\d+\.\d+',text) else '0')
        p._p.pPr.append(outline)
    p.paragraph_format.widow_control=True
    set_font(p.add_run(text),bold=kind in ['title','heading'])
    return p

def caption(text):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing=1.15; p.paragraph_format.space_after=Pt(8)
    set_font(p.add_run(text),size=11)
    return p

blocks=ROOT.joinpath('chapter_three.md').read_text(encoding='utf-8').split('\n\n')
in_refs=False
for block in blocks:
    block=block.strip()
    if not block:continue
    if block.startswith('# CHAPTER'):
        paragraph('CHAPTER THREE','title')
        paragraph('RESEARCH METHODOLOGY','title')
    elif block.startswith('## '):
        heading=block[3:]
        if heading=='REFERENCES':
            in_refs=True
            p=paragraph(heading,'heading'); p.paragraph_format.page_break_before=True
        else: paragraph(heading,'heading')
    elif block.startswith('### '):paragraph(block[4:],'heading')
    elif block.startswith('FIGURE:'):
        name,cap=[x.strip() for x in block[7:].split('|')]
        p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next=True; p.paragraph_format.space_before=Pt(6)
        p.paragraph_format.space_after=Pt(6); p.paragraph_format.line_spacing=1
        r=p.add_run(); r.add_picture(str(ROOT/(name+'.png')),width=Inches(6.15))
        inline=r._r.xpath('.//wp:docPr')[0]; inline.set('descr',cap)
        caption(cap)
    elif block.startswith('CAPTION:'):caption(block[8:].strip())
    elif block.startswith('|'):
        rows=[[c.strip() for c in l.strip('|').split('|')] for l in block.splitlines() if '---' not in l]
        table=doc.add_table(rows=0,cols=4); table.autofit=False
        widths=[.65,2.05,2.05,1.5]
        for rowid,row in enumerate(rows):
            cells=table.add_row().cells
            trpr=table.rows[-1]._tr.get_or_add_trPr()
            nosplit=OxmlElement('w:cantSplit');trpr.append(nosplit)
            if rowid==0:trpr.append(OxmlElement('w:tblHeader'))
            for i,(cell,txt) in enumerate(zip(cells,row)):
                cell.width=Inches(widths[i]); cell.vertical_alignment=1
                if i==0 and rowid==0:txt='No.'
                if txt=='Stacked bidirectional LSTM':txt='LSTM'
                p=cell.paragraphs[0];p.paragraph_format.line_spacing=1.05
                p.paragraph_format.space_after=Pt(1);p.paragraph_format.space_before=Pt(1)
                if i==0:p.alignment=WD_ALIGN_PARAGRAPH.CENTER
                set_font(p.add_run(txt),rowid==0,11)
                tcp=cell._tc.get_or_add_tcPr()
                mar=OxmlElement('w:tcMar')
                for side in ['top','left','bottom','right']:
                    el=OxmlElement('w:'+side);el.set(qn('w:w'),'50' if side in ['top','bottom'] else '85');el.set(qn('w:type'),'dxa');mar.append(el)
                tcp.append(mar)
                borders=OxmlElement('w:tcBorders')
                for side in ['top','left','bottom','right']:
                    el=OxmlElement('w:'+side);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'4');el.set(qn('w:color'),'D9D9D9');borders.append(el)
                tcp.append(borders)
                if rowid==0:
                    shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E7E6E6');tcp.append(shade)
        for i,w in enumerate(widths):table.columns[i].width=Inches(w)
    else:
        p=paragraph(block)
        if in_refs:
            p.alignment=WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.first_line_indent=Inches(-.5)
            p.paragraph_format.left_indent=Inches(.5)
            p.paragraph_format.space_after=Pt(8)
            p.paragraph_format.keep_together=True
            italics={
                'Devlin':'Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies',
                'Friedman':'Journal of Statistical Software, 33',
                'Gjurković':'Proceedings of the Ninth International Workshop on Natural Language Processing for Social Media',
                'Goodfellow':'Advances in Neural Information Processing Systems',
                'Hochreiter':'Neural Computation, 9',
                'Sutton':'Reinforcement learning: An introduction',
                'Tibshirani':'Journal of the Royal Statistical Society: Series B (Methodological), 58',
            }
            for prefix,span in italics.items():
                if block.startswith(prefix) and span in block:
                    p.runs[0].text=''
                    a,b=block.split(span,1)
                    set_font(p.add_run(a));r=p.add_run(span);set_font(r);r.italic=True;set_font(p.add_run(b))

doc.core_properties.title='Chapter Three Research Methodology'
doc.core_properties.subject='Personality prediction from PANDORA comments'
doc.core_properties.author=''
doc.core_properties.last_modified_by=''
settings=doc.settings.element
update=OxmlElement('w:updateFields');update.set(qn('w:val'),'true');settings.append(update)
doc.save(OUT)
# Preserve all untouched template parts exactly. Body/images/relationships/settings
# and document properties are the intended editable parts of this new chapter.
editable={'word/document.xml','word/_rels/document.xml.rels','[Content_Types].xml','word/settings.xml','docProps/core.xml','docProps/app.xml'}
with zipfile.ZipFile(SOURCE) as z:
    source_parts={n:z.read(n) for n in z.namelist()}
with zipfile.ZipFile(OUT) as z:
    output_parts={n:z.read(n) for n in z.namelist()}
for n,b in source_parts.items():
    if n not in editable and n in output_parts:output_parts[n]=b
with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED) as z:
    for n,b in output_parts.items():z.writestr(n,b)
inventory={n:{'size':len(b),'sha256':hashlib.sha256(b).hexdigest(),'editable':n in editable} for n,b in source_parts.items()}
ROOT.joinpath('source_inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
with (ROOT/'artifact.md').open('a',encoding='utf-8') as f:f.write('\nReference SHA256: '+hashlib.sha256(SOURCE.read_bytes()).hexdigest()+'\n')
print(OUT)
print('Body word count:',len(ROOT.joinpath('chapter_three.md').read_text(encoding='utf-8').split('## REFERENCES')[0].split()))
