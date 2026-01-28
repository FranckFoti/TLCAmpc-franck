import numpy as np
import pytest
from drone_sim.simulation.admm_state import ADMMState
from drone_sim.simulation.neighbor_graph import NeighborGraph


class TestADMMStateInitialization:
    """Tests for ADMMState initialization."""

    def test_initialize_creates_zero_dual_variables(self):
        """initialize() creates zero dual variables for all pairs."""
        state = ADMMState(horizon=5, rho=1.0)
        pairs = [("drone-1", "drone-2"), ("drone-1", "drone-3")]
        state.initialize(pairs)

        lam = state.get_lambda(("drone-1", "drone-2"))
        assert lam is not None
        assert lam.shape == (5, 3)
        assert np.allclose(lam, 0.0)

        z = state.get_z(("drone-1", "drone-2"))
        assert z is not None
        assert z.shape == (5, 3)
        assert np.allclose(z, 0.0)

    def test_initialize_clears_previous_state(self):
        """initialize() clears any previous state."""
        state = ADMMState(horizon=5, rho=1.0)
        state.initialize([("drone-1", "drone-2")])

        # Modify state
        traj1 = np.ones((5, 3)) * 10.0
        traj2 = np.zeros((5, 3))
        state.update_z(("drone-1", "drone-2"), traj1, traj2, 1.0)

        # Reinitialize should clear
        state.initialize([("drone-3", "drone-4")])

        # Old pair should not exist
        assert state.get_lambda(("drone-1", "drone-2")) is None
        assert state.get_z(("drone-1", "drone-2")) is None

        # New pair should be zero
        lam = state.get_lambda(("drone-3", "drone-4"))
        assert lam is not None
        assert np.allclose(lam, 0.0)

    def test_initialize_canonicalizes_pairs(self):
        """initialize() handles pair ordering correctly."""
        state = ADMMState(horizon=5)
        state.initialize([("b", "a")])  # Reversed order

        # Should still be accessible with either order
        assert state.get_lambda(("a", "b")) is not None
        assert state.get_lambda(("b", "a")) is not None


class TestADMMStateZUpdate:
    """Tests for z-update (projection)."""

    @pytest.fixture
    def state(self):
        """ADMMState with initialized pair."""
        s = ADMMState(horizon=5, rho=1.0)
        s.initialize([("drone-1", "drone-2")])
        return s

    def test_update_z_no_projection_when_satisfying_constraint(self, state):
        """update_z() with trajectories already satisfying constraint."""
        # Drones 10 units apart, min_dist is 1
        traj1 = np.array([[0, 0, 0]] * 5, dtype=float)
        traj2 = np.array([[10, 0, 0]] * 5, dtype=float)

        state.update_z(("drone-1", "drone-2"), traj1, traj2, min_dist=1.0)

        z = state.get_z(("drone-1", "drone-2"))
        expected = traj1 - traj2  # [-10, 0, 0]
        assert np.allclose(z, expected)

    def test_update_z_projects_when_violating_constraint(self, state):
        """update_z() with violating trajectories projects to min_dist."""
        # Drones 0.5 units apart, min_dist is 2.0
        traj1 = np.array([[0, 0, 0]] * 5, dtype=float)
        traj2 = np.array([[0.5, 0, 0]] * 5, dtype=float)

        state.update_z(("drone-1", "drone-2"), traj1, traj2, min_dist=2.0)

        z = state.get_z(("drone-1", "drone-2"))
        # z should be scaled to have norm = 2.0 in direction of (p1 - p2)
        for k in range(5):
            norm = np.linalg.norm(z[k])
            assert np.isclose(norm, 2.0, atol=1e-6)
            # Direction should be negative x (p1 - p2 = [-0.5, 0, 0])
            assert z[k, 0] < 0

    def test_update_z_stores_previous_z(self, state):
        """update_z() stores previous z for dual residual computation."""
        traj1 = np.array([[0, 0, 0]] * 5, dtype=float)
        traj2 = np.array([[5, 0, 0]] * 5, dtype=float)

        state.update_z(("drone-1", "drone-2"), traj1, traj2, min_dist=1.0)
        z_first = state.get_z(("drone-1", "drone-2")).copy()

        # Update again with different trajectories
        traj2_new = np.array([[8, 0, 0]] * 5, dtype=float)
        state.update_z(("drone-1", "drone-2"), traj1, traj2_new, min_dist=1.0)

        # prev_z should be the first z (indirectly tested via dual residual)
        z_second = state.get_z(("drone-1", "drone-2"))
        assert not np.allclose(z_first, z_second)

    def test_update_z_handles_zero_difference(self, state):
        """update_z() handles zero difference (same position)."""
        traj1 = np.array([[5, 5, 5]] * 5, dtype=float)
        traj2 = np.array([[5, 5, 5]] * 5, dtype=float)  # Same position

        state.update_z(("drone-1", "drone-2"), traj1, traj2, min_dist=2.0)

        z = state.get_z(("drone-1", "drone-2"))
        # Should project to arbitrary direction with norm = min_dist
        for k in range(5):
            norm = np.linalg.norm(z[k])
            assert np.isclose(norm, 2.0, atol=1e-6)


class TestADMMStateLambdaUpdate:
    """Tests for lambda-update (dual variable)."""

    @pytest.fixture
    def state(self):
        """ADMMState with initialized pair."""
        s = ADMMState(horizon=5, rho=2.0)  # rho=2 for easier testing
        s.initialize([("drone-1", "drone-2")])
        return s

    def test_update_lambda_increments_correctly(self, state):
        """update_lambda() increments dual variable correctly."""
        traj1 = np.array([[0, 0, 0]] * 5, dtype=float)
        traj2 = np.array([[3, 0, 0]] * 5, dtype=float)

        # First set z via update_z
        state.update_z(("drone-1", "drone-2"), traj1, traj2, min_dist=1.0)

        # Now update lambda
        state.update_lambda(("drone-1", "drone-2"), traj1, traj2)

        lam = state.get_lambda(("drone-1", "drone-2"))
        z = state.get_z(("drone-1", "drone-2"))
        diff = traj1 - traj2  # [-3, 0, 0]

        # lambda = 0 + rho * (diff - z)
        expected = state.rho * (diff - z)
        assert np.allclose(lam, expected)

    def test_update_lambda_accumulates(self, state):
        """Multiple lambda updates accumulate."""
        # Use trajectories where diff != z to create a residual
        # Drones 0.5 apart, but z will be projected to min_dist=2
        traj1 = np.array([[0, 0, 0]] * 5, dtype=float)
        traj2 = np.array([[0.5, 0, 0]] * 5, dtype=float)

        # z will be projected to norm=2, but diff is only 0.5
        # So diff - z != 0, causing lambda to accumulate
        state.update_z(("drone-1", "drone-2"), traj1, traj2, min_dist=2.0)

        # Multiple updates - each should add rho * (diff - z)
        state.update_lambda(("drone-1", "drone-2"), traj1, traj2)
        lam1 = state.get_lambda(("drone-1", "drone-2")).copy()

        state.update_lambda(("drone-1", "drone-2"), traj1, traj2)
        lam2 = state.get_lambda(("drone-1", "drone-2"))

        # Second should be roughly double (z unchanged between updates)
        assert np.linalg.norm(lam1) > 0  # First update creates non-zero lambda
        assert np.linalg.norm(lam2) > np.linalg.norm(lam1)


class TestADMMStateConsensusTerm:
    """Tests for consensus term computation."""

    @pytest.fixture
    def setup_state(self):
        """ADMMState with neighbor graph and initialized pair."""
        state = ADMMState(horizon=5, rho=1.0)
        graph = NeighborGraph(comm_radius=None)
        positions = {
            "drone-1": np.array([0, 0, 0]),
            "drone-2": np.array([5, 0, 0]),
        }
        graph.update(positions)
        state.initialize(graph.get_neighbor_pairs())
        return state, graph

    def test_get_consensus_term_returns_correct_shape(self, setup_state):
        """get_consensus_term() returns correct shape (H, 3)."""
        state, graph = setup_state
        trajectories = {
            "drone-1": np.zeros((5, 3)),
            "drone-2": np.ones((5, 3)) * 5,
        }

        term = state.get_consensus_term("drone-1", trajectories, graph)
        assert term.shape == (5, 3)

    def test_consensus_term_zero_when_satisfied_and_lambda_zero(self, setup_state):
        """Consensus term is zero when constraints satisfied and lambda=0."""
        state, graph = setup_state

        # Trajectories match z (which is zero initially)
        # p_i - p_j = z means residual is zero
        # lambda = 0 initially
        trajectories = {
            "drone-1": np.zeros((5, 3)),
            "drone-2": np.zeros((5, 3)),
        }

        term = state.get_consensus_term("drone-1", trajectories, graph)
        assert np.allclose(term, 0.0)

    def test_consensus_term_pushes_away_when_violated(self, setup_state):
        """Consensus term pushes drone away when constraint violated."""
        state, graph = setup_state

        # Drones too close - violation
        traj1 = np.array([[0, 0, 0]] * 5, dtype=float)
        traj2 = np.array([[0.5, 0, 0]] * 5, dtype=float)
        trajectories = {"drone-1": traj1, "drone-2": traj2}

        # Update z (will project to min_dist)
        state.update_z(("drone-1", "drone-2"), traj1, traj2, min_dist=2.0)
        state.update_lambda(("drone-1", "drone-2"), traj1, traj2)

        term1 = state.get_consensus_term("drone-1", trajectories, graph)
        term2 = state.get_consensus_term("drone-2", trajectories, graph)

        # Terms should push drones apart (opposite directions)
        # Since drone-1 is at x=0 and drone-2 at x=0.5, term1 should push negative
        # and term2 should push positive to increase separation
        assert term1[0, 0] != 0  # Non-zero consensus term
        # Signs should be opposite
        assert np.sign(term1[0, 0]) != np.sign(term2[0, 0])


class TestADMMStateResiduals:
    """Tests for residual computation."""

    @pytest.fixture
    def state(self):
        """ADMMState with initialized pair."""
        s = ADMMState(horizon=5, rho=1.0)
        s.initialize([("drone-1", "drone-2")])
        return s

    def test_compute_residuals_zero_at_initialization_with_matching_z(self, state):
        """compute_residuals() returns (0, 0) at init with matching trajectories."""
        # z is zero, so trajectories should also differ by zero for primal=0
        trajectories = {
            "drone-1": np.zeros((5, 3)),
            "drone-2": np.zeros((5, 3)),
        }

        primal, dual = state.compute_residuals(trajectories)
        assert np.isclose(primal, 0.0)
        assert np.isclose(dual, 0.0)

    def test_primal_residual_increases_with_violation(self, state):
        """Primal residual increases with constraint violation."""
        traj1 = np.array([[0, 0, 0]] * 5, dtype=float)
        traj2 = np.array([[1, 0, 0]] * 5, dtype=float)
        trajectories = {"drone-1": traj1, "drone-2": traj2}

        # z is still zero, but p_i - p_j != z
        primal, dual = state.compute_residuals(trajectories)
        assert primal > 0  # Should be > 0 since diff - z != 0

    def test_dual_residual_increases_when_z_changes(self, state):
        """Dual residual increases when z changes."""
        traj1 = np.zeros((5, 3), dtype=float)
        traj2 = np.array([[5, 0, 0]] * 5, dtype=float)

        # First update sets z
        state.update_z(("drone-1", "drone-2"), traj1, traj2, min_dist=1.0)

        # Second update with different trajectories changes z
        traj2_new = np.array([[10, 0, 0]] * 5, dtype=float)
        state.update_z(("drone-1", "drone-2"), traj1, traj2_new, min_dist=1.0)

        trajectories = {"drone-1": traj1, "drone-2": traj2_new}
        primal, dual = state.compute_residuals(trajectories)
        assert dual > 0  # z changed, so dual residual should be > 0


class TestADMMStateConvergence:
    """Tests for convergence checking."""

    def test_is_converged_returns_true_when_both_residuals_below_tolerance(self):
        """is_converged() returns True when both residuals below tolerance."""
        state = ADMMState(horizon=5, rho=1.0, primal_tol=1.0, dual_tol=1.0)
        state.initialize([("drone-1", "drone-2")])

        # Trajectories matching z (zero)
        trajectories = {
            "drone-1": np.zeros((5, 3)),
            "drone-2": np.zeros((5, 3)),
        }

        assert state.is_converged(trajectories) is True

    def test_is_converged_returns_false_when_primal_too_high(self):
        """is_converged() returns False when primal residual too high."""
        state = ADMMState(horizon=5, rho=1.0, primal_tol=0.1, dual_tol=1.0)
        state.initialize([("drone-1", "drone-2")])

        # Trajectories NOT matching z (zero) - large primal residual
        trajectories = {
            "drone-1": np.zeros((5, 3)),
            "drone-2": np.array([[5, 0, 0]] * 5, dtype=float),
        }

        assert state.is_converged(trajectories) is False

    def test_is_converged_returns_false_when_dual_too_high(self):
        """is_converged() returns False when dual residual too high."""
        state = ADMMState(horizon=5, rho=1.0, primal_tol=100.0, dual_tol=0.001)
        state.initialize([("drone-1", "drone-2")])

        traj1 = np.zeros((5, 3), dtype=float)
        traj2 = np.array([[5, 0, 0]] * 5, dtype=float)

        # First z update
        state.update_z(("drone-1", "drone-2"), traj1, traj2, min_dist=1.0)

        # Second z update with different traj - causes z change
        traj2_new = np.array([[10, 0, 0]] * 5, dtype=float)
        state.update_z(("drone-1", "drone-2"), traj1, traj2_new, min_dist=1.0)

        trajectories = {"drone-1": traj1, "drone-2": traj2_new}
        assert state.is_converged(trajectories) is False


class TestADMMStateIntegration:
    """Integration test for full ADMM cycle."""

    def test_full_admm_cycle(self):
        """Full ADMM cycle: initialize -> z-update -> lambda-update -> converge."""
        # Setup
        state = ADMMState(horizon=5, rho=1.0, primal_tol=0.01, dual_tol=0.01)
        graph = NeighborGraph(comm_radius=None)
        positions = {
            "drone-1": np.array([0, 0, 0]),
            "drone-2": np.array([5, 0, 0]),
        }
        graph.update(positions)
        state.initialize(graph.get_neighbor_pairs())

        # Initial trajectories - drones too close at timestep 0
        # They should be pushed apart through ADMM iterations
        min_dist = 2.0
        trajectories = {
            "drone-1": np.array([[0, 0, 0]] * 5, dtype=float),
            "drone-2": np.array([[1, 0, 0]] * 5, dtype=float),  # Only 1 unit apart
        }

        # Run ADMM iterations
        for iteration in range(10):
            # z-update for all pairs
            for pair in graph.get_neighbor_pairs():
                state.update_z(
                    pair,
                    trajectories[pair[0]],
                    trajectories[pair[1]],
                    min_dist,
                )

            # lambda-update for all pairs
            for pair in graph.get_neighbor_pairs():
                state.update_lambda(
                    pair,
                    trajectories[pair[0]],
                    trajectories[pair[1]],
                )

            # Check convergence
            primal, dual = state.compute_residuals(trajectories)

            # Note: In real ADMM, we'd update trajectories via local MPC
            # Here we just test the state updates work correctly

        # After iterations, dual variables should have accumulated
        lam = state.get_lambda(("drone-1", "drone-2"))
        assert lam is not None
        assert not np.allclose(lam, 0.0)  # Lambda should have accumulated

        # z should have been projected to satisfy min_dist
        z = state.get_z(("drone-1", "drone-2"))
        assert z is not None
        for k in range(5):
            norm = np.linalg.norm(z[k])
            assert norm >= min_dist - 1e-6

    def test_two_drones_converging_to_collision_free(self):
        """Two drones with ADMM consensus terms converging toward safe separation."""
        state = ADMMState(horizon=3, rho=0.5, primal_tol=0.1, dual_tol=0.1)
        graph = NeighborGraph(comm_radius=None)

        # Start with drones close together
        positions = {"d1": np.array([0, 0, 0]), "d2": np.array([0.5, 0, 0])}
        graph.update(positions)
        state.initialize(graph.get_neighbor_pairs())

        # Initial trajectories
        traj1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
        traj2 = np.array([[0.5, 0, 0], [0.5, 0, 0], [0.5, 0, 0]], dtype=float)
        trajectories = {"d1": traj1, "d2": traj2}

        min_dist = 2.0

        for _ in range(5):
            # z-update
            state.update_z(("d1", "d2"), trajectories["d1"], trajectories["d2"], min_dist)

            # lambda-update
            state.update_lambda(("d1", "d2"), trajectories["d1"], trajectories["d2"])

            # Get consensus terms (in real ADMM this would affect local MPC)
            term1 = state.get_consensus_term("d1", trajectories, graph)
            term2 = state.get_consensus_term("d2", trajectories, graph)

            # Terms should push drones apart
            # d1 should be pushed negative (away from d2 which is at +x)
            # d2 should be pushed positive (away from d1 which is at origin)
            assert term1.shape == (3, 3)
            assert term2.shape == (3, 3)

        # z should enforce minimum distance
        z = state.get_z(("d1", "d2"))
        for k in range(3):
            assert np.linalg.norm(z[k]) >= min_dist - 1e-6
