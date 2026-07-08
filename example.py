import numpy as np
from pysph.base.kernels import CubicSpline
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.equation import Equation, Group
from pysph.sph.integrator import EulerIntegrator
from pysph.sph.integrator_step import IntegratorStep

DEFAULT_N = 1024
DEFAULT_STEPS = 100
DEFAULT_DT = 1.0e-3
DIFFUSIVITY = 0.05


class DiffusePhi(Equation):
    def __init__(self, dest, sources):
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_aphi):
        d_aphi[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_phi, s_phi, d_aphi, s_m, s_rho, WIJ):
        d_aphi[d_idx] += DIFFUSIVITY * (s_phi[s_idx] - d_phi[d_idx]) * WIJ * s_m[s_idx] / s_rho[s_idx]


class MoveAndDiffuseStep(IntegratorStep):
    def stage1(self, d_idx, d_x, d_y, d_phi, d_u, d_v, d_aphi, dt):
        d_x[d_idx] += dt * d_u[d_idx]
        d_y[d_idx] += dt * d_v[d_idx]
        d_phi[d_idx] += dt * d_aphi[d_idx]


class MinimalPySPH(Application):
    def add_user_options(self, group):
        group.add_argument("--n", type=int, default=DEFAULT_N)
        group.add_argument("--steps", type=int, default=DEFAULT_STEPS)
        group.add_argument("--dt", type=float, default=DEFAULT_DT)

    def consume_user_options(self):
        self.n = self.options.n
        self.steps = self.options.steps
        self.dt = self.options.dt
        assert self.n >= 4
        assert self.steps >= 1
        assert self.dt > 0.0
        self.options.disable_output = True

    def create_particles(self):
        dx = 1.0 / self.n
        grid = (np.arange(self.n) + 0.5) * dx
        xg, yg = np.meshgrid(grid, grid)
        x = xg.ravel()
        y = yg.ravel()
        phi = np.exp(-80.0 * ((x - 0.35) ** 2 + (y - 0.5) ** 2))
        fluid = get_particle_array(name="fluid", x=x, y=y, h=1.4 * dx, m=dx * dx, rho=1.0, u=0.1, phi=phi, aphi=0.0)
        return [fluid]

    def create_solver(self):
        integrator = EulerIntegrator(fluid=MoveAndDiffuseStep())
        solver = Solver(kernel=CubicSpline(dim=2), dim=2, integrator=integrator, dt=self.dt, tf=self.steps * self.dt, fixed_h=True)
        solver.set_print_freq(self.steps)
        solver.set_disable_output(True)
        return solver

    def create_equations(self):
        return [Group(equations=[DiffusePhi(dest="fluid", sources=["fluid"])])]

    def solve(self):
        super().solve()
        fluid = self.particles[0]
        phi_l2 = np.linalg.norm(fluid.phi) / np.sqrt(fluid.phi.size)
        print(f"particles={fluid.get_number_of_particles(real=True)} steps={self.solver.count} phi_l2={phi_l2:.9g}")


if __name__ == "__main__":
    MinimalPySPH().run()
