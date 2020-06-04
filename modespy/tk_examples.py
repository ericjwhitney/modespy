#!/usr/bin/env python3
"""
This module provides a standalone Tkinter GUI app showing examples of how
MODES can be used, including direct integration with SciPy function
``solve_ivp()``.  Results are plotted using MatPlotLib which should be
installed before use.

Start the app by running this module from the command line:

``python -m modespy.tk_examples``
"""

# Last updated: 4 June 2020 by Eric J. Whitney

__all__ = ['main']

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
from scipy.integrate import solve_ivp
import sys
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.scrolledtext as scrolledtext
from typing import Any, Type

# noinspection PyUnresolvedReferences
from modespy import FILTERS, filter_order, METHODS, SolverType, MODES
from modespy.std_problems import *


# ----------------------------------------------------------------------------

class LabelledWidget(tk.Frame):
    """Convenience widget combining a Label and another widget formatted
    L-to-R with a predefined internal value. Any additional ``**kwargs``
    are passed to the main widget."""

    def __init__(self, parent, *, label: str, widget_type: Type,
                 val_type: Type[tk.Variable], start_value, **kwargs):
        super().__init__(parent)
        if label is not None:
            tk.Label(self, text=label).pack(side=tk.LEFT, padx=2, pady=2)
        self._value = val_type(value=start_value)
        self._widget = widget_type(self, justify=tk.CENTER,
                                   textvariable=self._value, **kwargs)
        self._widget.pack(side=tk.LEFT, padx=2, pady=2, expand=True, fill='x')

    @property
    def value(self):
        return self._value.get()

    @value.setter
    def value(self, x):
        self._value.set(x)


class LblCombo(LabelledWidget):
    """Labelled ComboBox widget adding a callback function when selected and a
    method for disabling / greying out."""

    def __init__(self, parent, *, label: str = None,
                 on_select: Callable = None, start_value=None, **kwargs):
        kwargs.setdefault('state', 'readonly')
        super().__init__(parent, label=label, widget_type=ttk.Combobox,
                         val_type=tk.StringVar, start_value=start_value,
                         **kwargs)
        self._widget.option_add('*TCombobox*Listbox.Justify', tk.CENTER)
        if on_select is not None:
            self._widget.bind("<<ComboboxSelected>>", on_select)

    def current(self, *args, **kwargs):
        return self._widget.current(*args, **kwargs)

    def enabled(self, on: bool = True):
        if on:
            self._widget.configure(state='readonly')
        else:
            self._widget.configure(state='disabled')


class LblEntry(LabelledWidget):
    """Labelled Entry widget adding a verification function called when the
    value is changed.  The value is highlighted if the function returns
    False."""

    def __init__(self, parent, *, label: str = None,
                 val_type: Type[tk.Variable], start_value=None,
                 verify: Callable[[Any], bool] = None):
        super().__init__(parent, label=label, widget_type=tk.Entry,
                         val_type=val_type, start_value=start_value)
        if verify is not None:
            self._verify = verify
            self._value.trace_add('write', self._check_value)
        self._bg_normal = self._widget.cget('bg')
        self._bg_grey = parent.master.cget('bg')
        self._bg_hilite = 'yellow'

    def enabled(self, on: bool = True):
        if on:
            self._widget.configure(background=self._bg_normal, state='normal')
        else:
            self._widget.configure(background=self._bg_grey, state='readonly')

    # noinspection PyUnusedLocal
    def _check_value(self, var, idx, mode):
        if self._verify(self._value):
            self._widget.configure(background=self._bg_normal)
        else:
            self._widget.configure(background=self._bg_hilite)


# ----------------------------------------------------------------------------

class MODESExampleGUI(tk.Frame):
    """Main UI for running MODES problem examples."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.master.title('MODES Examples')
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)
        self.grid(sticky='nsew')
        self.col_bg_grey = self.master.cget('bg')

        self.problems = list(PROBLEMS.keys())

        self.methods = []
        for name, meth in METHODS.items():
            if meth.solver_type == SolverType.EXPLICIT:
                self.methods += [f"{name} (Exp)"]
            elif meth.solver_type == SolverType.IMPLICIT:
                self.methods += [f"{name} (Imp)"]
            else:
                self.methods += [f"{name} (Imp+1)"]

        self.filters = []
        for s in FILTERS.keys():
            if s is not None:
                self.filters.append(f"{s} ({filter_order(FILTERS[s])} Step)")
            else:
                self.filters.append(f"{s} (---)")

        # Problem Selection / Starting Values --------------------------------

        inp_frame = tk.Frame(self)
        inp_frame.grid(row=0, column=0, padx=2, pady=2)

        row = RowMonitor()
        self.problem_box = LblCombo(
            inp_frame, label='Problem', values=self.problems,
            on_select=self.change_problem, start_value=self.problems[0])
        self.problem_box.grid(row=row.current, column=0, columnspan=4)
        self.current_problem = None  # Set later by change_problem().

        self.descr_txt = scrolledtext.ScrolledText(
            inp_frame, height=4, width=40, padx=2, pady=2, wrap=tk.WORD,
            state=tk.DISABLED)
        self.descr_txt.grid(row=row.advance, column=0, columnspan=4,
                            sticky='nsew')

        self.t0_ent = LblEntry(inp_frame, label='t Start',
                               val_type=tk.DoubleVar, verify=check_tkvar)
        self.t0_ent.grid(row=row.advance, column=0, columnspan=2),

        self.tf_ent = LblEntry(inp_frame, label='t End',
                               val_type=tk.DoubleVar, verify=check_tkvar)
        self.tf_ent.grid(row=row.current, column=2, columnspan=2)

        self.y0_ent = LblEntry(inp_frame, label='y0', val_type=tk.StringVar,
                               verify=check_floatlist)
        self.y0_ent.grid(row=row.advance, column=0, columnspan=4)

        self.tol_ent = LblEntry(inp_frame, label='Tolerance',
                                val_type=tk.DoubleVar, start_value='1e-4',
                                verify=check_tkvar)
        self.tol_ent.grid(row=row.advance, column=0, columnspan=4)

        # Run Parameters - Common Solver -------------------------------------

        ttk.Separator(inp_frame, orient=tk.HORIZONTAL).grid(
            row=row.advance, columnspan=4, pady=5, sticky='ew')

        self.cmn_sel = tk.IntVar(value=1)
        self.cmn_chk = tk.Checkbutton(inp_frame,
                                      text="Use Common Method & Filter",
                                      takefocus=False, variable=self.cmn_sel,
                                      command=self.cmn_chk_clicked)
        self.cmn_chk.grid(row=row.advance, column=0, columnspan=4,
                          sticky='nsew')

        self.method_box = LblCombo(
            inp_frame, label='Method', values=self.methods,
            on_select=self.change_method, start_value='Adams-Moulton (Imp+1)')
        self.method_box.grid(row=row.advance, column=0, columnspan=4,
                             sticky='nsew')

        self.filter_box = LblCombo(
            inp_frame, label='Step Control Filter', values=self.filters,
            start_value='PI3333 (2 Step)')
        self.filter_box.grid(row=row.advance, column=0, columnspan=4,
                             sticky='nsew')

        self.pmin_ent = LblEntry(inp_frame, label='Order Range From',
                                 val_type=tk.IntVar, verify=check_tkvar)
        self.pmin_ent.grid(row=row.advance, column=0, columnspan=2,
                           sticky='nsew')
        self.pmax_ent = LblEntry(inp_frame, label='To',
                                 val_type=tk.IntVar, verify=check_tkvar)
        self.pmax_ent.grid(row=row.current, column=2, columnspan=2,
                           sticky='nsew')

        # Run Parameters - Individual Solvers --------------------------------

        ttk.Separator(inp_frame, orient=tk.HORIZONTAL).grid(
            row=row.advance, columnspan=4, pady=5, sticky='ew')

        tk.Label(inp_frame, text='Individual Solvers').grid(
            row=row.advance, column=0, columnspan=4, sticky='nsew')

        tk.Label(inp_frame, text='Order (or Blank)').grid(
            row=row.advance, column=0, sticky='nsew')
        tk.Label(inp_frame, text='Method').grid(
            row=row.current, column=1, columnspan=2, sticky='nsew')
        tk.Label(inp_frame, text='Filter').grid(
            row=row.current, column=3, sticky='nsew')

        nindiv = 8
        self.ind_p_ent, self.ind_m_box, self.ind_f_box = [], [], []
        for i in range(nindiv):
            self.ind_p_ent.append(LblEntry(
                inp_frame, val_type=tk.IntVar, start_value=i + 1,
                verify=lambda x: check_tkvar(x, allow_blank=True)))
            self.ind_p_ent[-1].grid(row=row.advance, column=0)
            self.ind_m_box.append(LblCombo(
                inp_frame, values=self.methods,
                start_value=self.method_box.value))
            self.ind_m_box[-1].grid(row=row.current, column=1, columnspan=2)
            self.ind_f_box.append(LblCombo(
                inp_frame, values=self.filters,
                start_value=self.filter_box.value))
            self.ind_f_box[-1].grid(row=row.current, column=3)

        # Buttons ------------------------------------------------------------

        ttk.Separator(inp_frame, orient=tk.HORIZONTAL).grid(
            row=row.advance, columnspan=4, pady=5, sticky='ew')

        self.verbose = tk.IntVar(value=1)
        tk.Checkbutton(inp_frame, text="Verbose Output", takefocus=False,
                       variable=self.verbose).grid(row=row.advance, column=0,
                                                   columnspan=4, sticky='nsew')

        self.run_btn = tk.Button(inp_frame, text='RUN MODES', fg='white',
                                 bg='blue', command=self.run_ode)
        self.run_btn.grid(row=row.advance, column=0, columnspan=4)

        # Plot Area ----------------------------------------------------------

        self.sol_fig = plt.Figure(figsize=(6, 6))
        self.sol_cvs = FigureCanvasTkAgg(self.sol_fig, master=self)
        self.sol_cvs.get_tk_widget().grid(row=0, column=1, sticky='nsew',
                                          padx=2, pady=2)

        # Status Text Box ----------------------------------------------------

        tk.Label(self, text="Output", anchor=tk.CENTER).grid(
            row=2, column=0, columnspan=2, sticky='nsew')
        self.status_txt = scrolledtext.ScrolledText(
            self, height=15, background=self.col_bg_grey, padx=2,
            pady=2, wrap=tk.WORD)
        self.status_txt.grid(row=3, column=0, columnspan=2, sticky='nsew')
        sys.stdout = TextRedirector(self.status_txt, 'stdout')
        print("--- NIL ---")  # Redirects to status window.

        self.columnconfigure(1, weight=1)
        self.rowconfigure([0, 1, 3], weight=1)
        self.change_problem(None)  # Init. default fields.
        self.change_method(None)
        self.cmn_chk_clicked()

    # noinspection PyUnusedLocal
    def change_method(self, event):
        m_str = list(METHODS)[self.method_box.current()]
        p_defaults = METHODS[m_str].p_defaults
        self.pmin_ent.value = p_defaults[0]
        self.pmax_ent.value = p_defaults[1]

    # noinspection PyUnusedLocal
    def change_problem(self, event):
        self.current_problem = PROBLEMS[self.problem_box.value]
        self.descr_txt.configure(state=tk.NORMAL)
        self.descr_txt.delete('1.0', tk.END)
        self.descr_txt.insert(tk.END, self.current_problem.description)
        self.descr_txt.configure(state=tk.DISABLED)

        if self.current_problem.default_t is not None:
            self.t0_ent.value = self.current_problem.default_t[0]
            self.tf_ent.value = self.current_problem.default_t[1]
        if self.current_problem.default_y0 is not None:
            self.y0_ent.value = ', '.join(str(e) for e in
                                          self.current_problem.default_y0)

    def clear_status(self):
        self.status_txt.configure(state=tk.NORMAL)
        self.status_txt.delete('1.0', tk.END)
        self.status_txt.configure(state=tk.DISABLED)

    def cmn_chk_clicked(self):
        select_cmn = bool(self.cmn_sel.get())
        for widg in (self.method_box, self.filter_box, self.pmin_ent,
                     self.pmax_ent):
            widg.enabled(select_cmn)
        for widg in (*self.ind_p_ent, *self.ind_m_box, *self.ind_f_box):
            widg.enabled(not select_cmn)

    def run_ode(self):
        self.clear_status()

        def make_filter_arg(filt_arg):  # Helper function for odd cases.
            if filt_arg is None:
                return "None"
            elif filt_arg == 'H211b':
                return f"('H211b', 4)"
            elif filt_arg == 'H312b':
                return f"('H312b', 8)"
            else:
                return f"'{filt_arg}'"

        # Build argument list for solve_ivp().
        try:
            t_span = [self.t0_ent.value, self.tf_ent.value]
            y0 = str2floatlist(self.y0_ent.value)
            tol = self.tol_ent.value
        except (tk.TclError, ValueError):
            print("Invalid Inputs.")
            return

        ivp_args = [f"fun={self.current_problem.equation.__name__}",
                    f"t_span={t_span}", f"y0={y0}", f"tol={tol}",
                    "method=MODES"]

        if self.cmn_sel.get():
            # All solvers / filters are common.
            ivp_args += [f"modes_p=({self.pmin_ent.value},"
                         f"{self.pmax_ent.value})"]
            use_method = list(METHODS)[self.method_box.current()]
            ivp_args += [f"modes_method='{use_method}'"]
            use_filter = list(FILTERS)[self.filter_box.current()]
            ivp_args += [f"modes_filter={make_filter_arg(use_filter)}"]
        else:
            # All solvers are unique.
            use_p, use_method, use_filter = [], [], []
            for p_w, m_w, f_w in zip(self.ind_p_ent, self.ind_m_box,
                                     self.ind_f_box):
                try:
                    use_p.append(str(p_w.value))
                    use_method.append(f"'{list(METHODS)[m_w.current()]}'")
                    use_filter.append(make_filter_arg(
                        list(FILTERS)[f_w.current()]))
                except tk.TclError:
                    pass  # Skip blank entries.

            ivp_args += [f"modes_p=({','.join(use_p)})"]
            ivp_args += [f"modes_method=({','.join(use_method)})"]
            ivp_args += [f"modes_filter=({','.join(use_filter)})"]

        if self.current_problem.jacobian:
            ivp_args += [f"jac={self.current_problem.jacobian.__name__}"]

        ivp_args += ["modes_stats=modes_stats"]

        if self.verbose.get():
            ivp_args += ["modes_config={'verbose': True}"]

        # Run MODES and show output.
        modes_stats = {}
        ivp_call = 'solve_ivp(' + ', '.join(ivp_args) + ')'

        print(f"Command:\n")
        print(f">>> {ivp_call}\n")
        print(f"Output:\n")

        sol = eval(ivp_call)

        print(f"Completed:")
        print(f"... Exit status: {sol.status}")
        print(f"... Message: {sol.message}")
        print(f"... Number of Function Evaluations: {sol.nfev}")
        print(f"... Number of Jacobian Evaluations: {sol.njev}")

        # Plot results.
        self.sol_fig.clf()
        sol_plt = self.sol_fig.subplots(4, 1, sharex='all', gridspec_kw={
            'height_ratios': [2, 1, 1, 1]})
        self.sol_fig.subplots_adjust(left=0.15, right=0.85, bottom=0.05,
                                     top=0.95, wspace=0.2)
        labels = []
        for i, y in enumerate(sol.y):
            sol_plt[0].plot(sol.t, y, '.-')
            labels.append("$y_{" + str(i) + "}$")
        if len(labels) <= 6:
            sol_plt[0].legend(labels)
        sol_plt[0].set_xlabel('$t$')
        sol_plt[0].set_ylabel('$y$')

        sol_plt[1].plot(modes_stats['t'], modes_stats['h'], 'k.-')
        sol_plt[1].set_ylabel('$h$')
        sol_plt[1].set_yscale('log')

        sol_plt[2].plot(modes_stats['t'], modes_stats['err_norm'], 'k.-')
        sol_plt[2].set_ylabel('$||E||$')
        sol_plt[2].set_yscale('log')

        sol_plt[3].plot(modes_stats['t'], modes_stats['p'], 'k.-')
        sol_plt[3].set_ylabel('$p$')
        sol_plt[3].yaxis.set_ticks(np.arange(1, max(modes_stats['p']) + 1, 1))

        for this_plt in sol_plt:
            this_plt.grid()
        self.sol_cvs.draw()


# ----------------------------------------------------------------------------


class TextRedirector:
    """Class allowing redirection of stdout to a Tkinter text widget.  Adapted
    from this StackOverflow answer: https://stackoverflow.com/a/12352237"""

    def __init__(self, widget, tag='stdout'):
        self.widget = widget
        self.tag = tag

    def write(self, wstr):
        self.widget.configure(state='normal')
        self.widget.insert(tk.END, wstr, (self.tag,))
        self.widget.configure(state='disabled')
        self.widget.see(tk.END)
        self.widget.update_idletasks()

    def flush(self):
        pass


class RowMonitor:
    def __init__(self, current=0):
        self.current = current

    @property
    def advance(self):
        self.current += 1
        return self.current


# ----------------------------------------------------------------------------

def check_tkvar(val: tk.Variable, allow_blank: bool = False):
    """Returns True if calling ``val.get()`` doesn't raise `TclError`."""
    try:
        val.get()
        return True
    except tk.TclError:
        # noinspection PyProtectedMember
        if allow_blank and not val._tk.globalgetvar(val._name).strip():
            return True
        else:
            return False


def check_floatlist(strvar: tk.StringVar):
    """Returns True if `strvar` contains delimited list of floats."""
    try:
        str2floatlist(strvar.get())
        return True
    except ValueError:
        return False


def str2floatlist(floats_str: str):
    return [float(x) for x in floats_str.split(',')]


# ----------------------------------------------------------------------------

def main():
    root = tk.Tk()
    MODESExampleGUI()
    root.mainloop()


if __name__ == "__main__":
    main()
