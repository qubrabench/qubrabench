from plotting_strategy import PlottingStrategy


class HillClimberPlottingStrategy(PlottingStrategy):
    def __init__(self):
        self.quantum_factor = 2

    def get_x_label(self):
        return "$n$"

    def get_y_label(self):
        return "Queries"

    def get_parameters_to_group_by(self):
        return "k", "r"

    def make_history_adjustments(self, history):
        # compute combined query costs of quantum search
        c = history["quantum_expected_classical_queries"]
        q = history["quantum_expected_quantum_queries"]
        history["quantum_cqq"] = c + self.quantum_factor * q
        return history

    def get_line_plotting_dict(self):
        return {
            "classical_actual_queries": "Classical Queries",
            "quantum_cqq": "Quantum Queries",
        }

    def get_plotted_column_name_list(self):
        return [
            "classical_actual_queries",
            "classical_expected_queries",
            "quantum_expected_classical_queries",
            "quantum_expected_quantum_queries",
        ]

    def calculate_means(self, impl):
        return impl.groupby("n").mean(numeric_only=True)

    def calculate_errors(self, impl):
        return impl.groupby("n").sem(numeric_only=True)

    def get_plot_point_symbol(self, label):
        return "x" if "Quantum" in label else "o"
