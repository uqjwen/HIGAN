from docx import Document
from docx.shared import Inches
document = Document()
document.add_heading('Document Title', 0)  #插入标题
p = document.add_paragraph('A plain paragraph having some ')   #插入段落
p.add_run('bold').bold = True
p.add_run(' and some ')
p.add_run('italic.').italic = True