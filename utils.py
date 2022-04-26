"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
import torch  # for torch based ops

DEVICE = lambda: "cuda" if torch.cuda.is_available() else "cpu"
