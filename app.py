import tkinter as tk
from tkinter import ttk, messagebox, Canvas, Frame, Scrollbar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from main import StatisticalAnalysis, BackTest, Index  # Import your existing logic
import pandas as pd


class BacktestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pair Trading Backtester")
        self.root.geometry("900x700")

        # Create a Scrollable Frame
        self.canvas = Canvas(root)
        self.scroll_frame = Frame(self.canvas)
        self.scrollbar = Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.scroll_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Index Selection
        ttk.Label(self.scroll_frame, text="Select Index:").pack()
        self.index_var = tk.StringVar()
        self.index_dropdown = ttk.Combobox(self.scroll_frame, textvariable=self.index_var,
                                           values=["djia", "sandp", "crypto"], state="readonly")
        self.index_dropdown.pack()
        self.index_dropdown.bind("<<ComboboxSelected>>", self.generate_pairs)

        # Optimal Pairs Selection
        ttk.Label(self.scroll_frame, text="Select Pair:").pack()
        self.pair_var = tk.StringVar()
        self.pair_dropdown = ttk.Combobox(self.scroll_frame, textvariable=self.pair_var, state="readonly")
        self.pair_dropdown.pack()

        # Run Button
        self.run_button = ttk.Button(self.scroll_frame, text="Run Backtest", command=self.run_backtest)
        self.run_button.pack(pady=10)

        # Result Display
        self.result_label = ttk.Label(self.scroll_frame, text="")
        self.result_label.pack()

        # Graph Frame
        self.graph_frame = ttk.Frame(self.scroll_frame)
        self.graph_frame.pack(pady=10)

        # Graph Buttons
        self.graphs_frame = ttk.Frame(self.scroll_frame)
        self.graphs_frame.pack(pady=10)
        ttk.Button(self.graphs_frame, text="Equity Curve", command=self.plot_pnl).grid(row=0, column=0, padx=5)
        ttk.Button(self.graphs_frame, text="Pair Prices", command=self.plot_pair_prices).grid(row=0, column=1, padx=5)
        ttk.Button(self.graphs_frame, text="Residuals", command=self.plot_residuals).grid(row=1, column=0, padx=5)
        ttk.Button(self.graphs_frame, text="Z-Score", command=self.plot_zscore).grid(row=1, column=1, padx=5)

        # Trades Table
        self.trades_table = ttk.Treeview(self.scroll_frame, columns=("Date", "Capital", "PnL"), show="headings")
        self.trades_table.heading("Date", text="Date")
        self.trades_table.heading("Capital", text="Capital ($)")
        self.trades_table.heading("PnL", text="PnL ($)")
        self.trades_table.pack()

    def generate_pairs(self, event):
        index_name = self.index_var.get()
        index = Index(index_name)
        stat = StatisticalAnalysis('stock data', '2024-01-01', '2025-01-01')
        optimal_pairs = stat.optimalPairs()
        self.pairs = [f"{row['Pair'][0]} - {row['Pair'][1]}" for _, row in optimal_pairs.iterrows()]
        self.pair_dropdown["values"] = self.pairs

    def run_backtest(self):
        selected_pair = self.pair_var.get()
        if not selected_pair:
            messagebox.showerror("Input Error", "Please select a pair to backtest.")
            return

        stock1, stock2 = selected_pair.split(" - ")
        try:
            stat = StatisticalAnalysis('stock data', '2024-01-01', '2025-01-01')
            self.bt = BackTest(stock1, stock2, stat.stockData, zentry=0.15, zexit=0, initial_capital=100000,
                               risk_per_trade=0.5)
            final_capital, _ = self.bt.run()
            self.result_label.config(text=f"Final Capital: ${final_capital:.2f}")
            self.display_trades()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def display_graph(self, fig):
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def plot_pnl(self):
        if not hasattr(self, 'bt') or not self.bt.pnl_history:
            messagebox.showerror("Error", "Please run a backtest first.")
            return
        fig = self.bt.plot_pnl()
        if fig:
            self.display_graph(fig)

    def plot_pair_prices(self):
        selected_pair = self.pair_var.get()
        if not selected_pair:
            messagebox.showerror("Input Error", "Please select a pair first.")
            return
        stock1, stock2 = selected_pair.split(" - ")
        stat = StatisticalAnalysis('stock data', '2024-01-01', '2025-01-01')
        fig = stat.plot_pair_prices(stock1, stock2)
        if fig:
            self.display_graph(fig)

    def plot_residuals(self):
        selected_pair = self.pair_var.get()
        if not selected_pair:
            messagebox.showerror("Input Error", "Please select a pair first.")
            return
        stock1, stock2 = selected_pair.split(" - ")
        stat = StatisticalAnalysis('stock data', '2024-01-01', '2025-01-01')
        fig = stat.plot_residuals(stock1, stock2)
        if fig:
            self.display_graph(fig)

    def plot_zscore(self):
        selected_pair = self.pair_var.get()
        if not selected_pair:
            messagebox.showerror("Input Error", "Please select a pair first.")
            return
        stock1, stock2 = selected_pair.split(" - ")
        stat = StatisticalAnalysis('stock data', '2024-01-01', '2025-01-01')
        fig = stat.plot_zscore(stock1, stock2)
        if fig:
            self.display_graph(fig)

    def display_trades(self):
        for i in self.trades_table.get_children():
            self.trades_table.delete(i)

        if not hasattr(self, 'bt') or not self.bt.pnl_history:
            return

        for date, capital, pnl in self.bt.pnl_history:
            formatted_date = pd.to_datetime(date).strftime('%Y-%m-%d')  # Format date properly
            self.trades_table.insert("", "end", values=(formatted_date, f"{capital:.2f}", f"{pnl:.2f}"))


if __name__ == "__main__":
    root = tk.Tk()
    app = BacktestApp(root)
    root.mainloop()
