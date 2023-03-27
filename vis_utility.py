from ipywidgets import *


def layout_generator(header, children, num_cols=2, col_width=200, row_height=40):
    num_rows = len(children) // num_cols
    aux_cols = 1
    if len(children) % num_cols:
        footer = children[-1]
        aux_cols += 1
    else: footer = None
    
    grid = GridspecLayout(num_rows+aux_cols, num_cols, width=f"{col_width*num_cols}px", height=f'{row_height*(num_rows+aux_cols)}px', merge=True)
    grid[0, :] = header
    for row in range(num_rows):
         for col in range(num_cols):
            grid[row+1, col] = children[row*num_cols+col]
    if footer: 
        for col in range(num_cols):
            if num_rows*num_cols + col < len(children): grid[-1, col] = children[num_rows*num_cols + col]
    return grid
