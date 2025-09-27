#helpful scripts to use for all projects

import numpy as np
import skimage as sk
import skimage.io as skio
import os

def save_output_im(im_out, fname):  
    """Saves an image and outputs it

    Inputs:
        im_out: Stacked image in (H x W x 3) (r g b)
        fname name of file
    """
    # save the image
    fname = f'../images/{fname}.jpg'
    if im_out.dtype == np.float64:
        im_out = (im_out * 255).astype(np.uint8)
    else:
        raise ValueError(f"Invalid image type {im_out.dtype}")
    skio.imsave(fname,im_out)
    
    
import re
import sys
import nbformat
from nbformat.v4 import new_markdown_cell

def slugify(text):
    """
    Turn "My Heading!" → "my-heading"
    (simple: lower, replace spaces/punctuation with dashes)
    """
    slug = text.lower()
    slug = re.sub(r'[^\w\s-]', '', slug)    # drop punctuation
    slug = re.sub(r'[\s_]+', '-', slug)      # spaces/underscores → dash
    return slug

def update_toc(nb_path):
    nb = nbformat.read(nb_path, as_version=4)

    # 1) collect all ### and #### headings with their hierarchy
    toc_items = []
    for cell in nb.cells:
        if cell.cell_type != 'markdown':
            continue
        for line in cell.source.splitlines():
            line = line.strip()
            if line.startswith("### "):
                title = line.lstrip("# ").strip()
                anchor = slugify(title)
                toc_items.append(('h3', title, anchor))
            elif line.startswith("#### "):
                title = line.lstrip("# ").strip()
                anchor = slugify(title)
                toc_items.append(('h4', title, anchor))

    # 2) build the hierarchical ToC markdown
    toc_lines = ["# Table of Contents\n"]
    
    for level, title, anchor in toc_items:
        if level == 'h3':
            # Main heading - no indentation
            toc_lines.append(f"- [{title}](#{anchor})")
        elif level == 'h4':
            # Subheading - indented with two spaces
            toc_lines.append(f"  - [{title}](#{anchor})")
    
    toc_md = "\n".join(toc_lines)
    toc_cell = new_markdown_cell(toc_md)

    # 3) insert/replace at index 2 (third position)
    if len(nb.cells) >= 3:
        nb.cells[2] = toc_cell
    else:
        nb.cells.insert(2, toc_cell)

    # 4) write back
    nbformat.write(nb, nb_path)
    print(f"✅ Updated hierarchical ToC in {nb_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_toc.py notebook.ipynb")
        sys.exit(1)
    update_toc(sys.argv[1])