#@Author : tmayer, FGuillaume, RGaspard, Daniel_JK
from src.UI.ui import CVRPApp
import tkinter as tk

def main():
    """
    Main launch of CVRP solution evaluation using Savings, ACO Colony, and Genetic Algorithm.
    """
    root = tk.Tk()
    app = CVRPApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
