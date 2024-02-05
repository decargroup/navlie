from typing import Any, List
import numpy as np
from navlie import State, Measurement, MeasurementModel, ProcessModel, Input
from scipy.linalg import block_diag


class CompositeState(State):
    """
    A "composite" state object intended to hold a list of State objects as a
    single conceptual "state". The intended use is to hold a list of states
    as a single state at a specific time, of potentially different types,
    and this class will take care of defining the appropriate operations on
    the composite state such as the ``plus`` and ``minus`` methods, as well
    as the ``plus_jacobian`` and ``minus_jacobian`` methods.

    Each state in the provided list has an index (the index in the list), as
    well as a state_id, which is found as an attribute in the corresponding State
    object.

    It is possible to access sub-states in the composite states both by index
    and by ID.
    """

    def __init__(
        self, state_list: List[State], stamp: float = None, state_id=None
    ):
        """
        Parameters
        ----------
        state_list: List[State]
            List of State that forms this composite state
        stamp: float, optional
            Timestamp of the composite state. This can technically be different
            from the timestamps of the substates.
        state_id: Any, optional
            State ID of the composite state. This can be different from the
            state IDs of the substates.
        """
        if isinstance(state_list, tuple):
            state_list = list(state_list)

        #:List[State]: The substates are the CompositeState's value.
        self.value = state_list

        self.stamp = stamp
        self.state_id = state_id

    def __getstate__(self):
        """
        Get the state of the object for pickling.
        """
        # When using __slots__ the pickle module expects a tuple from __getstate__.
        # See https://stackoverflow.com/questions/1939058/simple-example-of-use-of-setstate-and-getstate/41754104#41754104
        return (
            None,
            {
                "value": self.value,
                "stamp": self.stamp,
                "state_id": self.state_id,
            },
        )

    def __setstate__(self, attributes):
        """
        Set the state of the object for unpickling.
        """
        # When using __slots__ the pickle module sends a tuple for __setstate__.
        # See https://stackoverflow.com/questions/1939058/simple-example-of-use-of-setstate-and-getstate/41754104#41754104

        attributes = attributes[1]
        self.value = attributes["value"]
        self.stamp = attributes["stamp"]
        self.state_id = attributes["state_id"]

    @property
    def dof(self):
        return sum([x.dof for x in self.value])

    def get_index_by_id(self, state_id):
        """
        Get index of a particular state_id in the list of states.
        """
        return [x.state_id for x in self.value].index(state_id)

    def get_slices(self) -> List[slice]:
        """
        Get slices for each state in the list of states.
        """
        slices = []
        counter = 0
        for state in self.value:
            slices.append(slice(counter, counter + state.dof))
            counter += state.dof

        return slices

    def add_state(self, state: State, stamp: float = None, state_id=None):
        """Adds a state and it's corresponding slice to the composite state."""
        self.value.append(state)

    def remove_state_by_id(self, state_id):
        """Removes a given state by ID."""
        idx = self.get_index_by_id(state_id)
        self.value.pop(idx)

    def get_slice_by_id(self, state_id, slices=None):
        """
        Get slice of a particular state_id in the list of states.
        """

        if slices is None:
            slices = self.get_slices()

        idx = self.get_index_by_id(state_id)
        return slices[idx]

    def get_matrix_block_by_ids(
        self, mat: np.ndarray, state_id_1: Any, state_id_2: Any = None
    ) -> np.ndarray:
        """Gets the portion of a matrix corresponding to two states.

        This function is useful when extract specific blocks of a covariance
        matrix, for example.

        Parameters
        ----------
        mat : np.ndarray
            N x N matrix
        state_id_1 : Any
            State ID of state 1.
        state_id_2 : Any, optional
            State ID of state 2. If None, state_id_2 is set to state_id_1.

        Returns
        -------
        np.ndarray
            Subblock of mat corrsponding to
            slices of state_id_1 and state_id_2.
        """

        if state_id_2 is None:
            state_id_2 = state_id_1

        slice_1 = self.get_slice_by_id(state_id_1)
        slice_2 = self.get_slice_by_id(state_id_2)

        return mat[slice_1, slice_2]

    def set_matrix_block_by_ids(
        self,
        new_mat_block: np.ndarray,
        mat: np.ndarray,
        state_id_1: Any,
        state_id_2: Any = None,
    ) -> np.ndarray:
        """Sets the portion of the covariance block corresponding to two states.

        Parameters
        ----------
        new_mat_block : np.ndarray
            A subblock to be entered into mat.
        mat : np.ndarray
            Full matrix.
        state_id_1 : Any
            State ID of state 1.
        state_id_2 : Any, optional
            State ID of state 2. If None, state_id_2 is set to state_id_1.

        Returns
        -------
        np.ndarray
            mat with updated subblock.
        """

        if state_id_2 is None:
            state_id_2 = state_id_1

        slice_1 = self.get_slice_by_id(state_id_1)
        slice_2 = self.get_slice_by_id(state_id_2)

        mat[slice_1, slice_2] = new_mat_block
        return mat

    def get_value_by_id(self, state_id) -> Any:
        """
        Get state value by id.
        """
        idx = self.get_index_by_id(state_id)
        return self.value[idx].value

    def get_state_by_id(self, state_id) -> State:
        """
        Get state object by id.
        """
        idx = self.get_index_by_id(state_id)
        return self.value[idx]

    def get_dof_by_id(self, state_id) -> int:
        """
        Get degrees of freedom of sub-state by id.
        """
        idx = self.get_index_by_id(state_id)
        return self.value[idx].dof

    def get_stamp_by_id(self, state_id) -> float:
        """
        Get timestamp of sub-state by id.
        """
        idx = self.get_index_by_id(state_id)
        return self.value[idx].stamp

    def set_stamp_by_id(self, stamp: float, state_id):
        """
        Set the timestamp of a sub-state by id.
        """
        idx = self.get_index_by_id(state_id)
        self.value[idx].stamp = stamp

    def set_state_by_id(self, state: State, state_id):
        """
        Set the whole sub-state by id.
        """
        idx = self.get_index_by_id(state_id)
        self.value[idx] = state

    def set_value_by_id(self, value: Any, state_id: Any):
        """
        Set the value of a sub-state by id.
        """
        idx = self.get_index_by_id(state_id)
        self.value[idx].value = value

    def set_stamp_for_all(self, stamp: float):
        """
        Set the timestamp of all substates.
        """
        for state in self.value:
            state.stamp = stamp

    def to_list(self):
        """
        Converts the CompositeState object back into a list of states.
        """
        return self.value

    def copy(self) -> "CompositeState":
        """
        Returns a new composite state object where the state values have also
        been copied.
        """
        return self.__class__(
            [state.copy() for state in self.value], self.stamp, self.state_id
        )

    def plus(self, dx, new_stamp: float = None) -> "CompositeState":
        """
        Updates the value of each sub-state given a dx. Interally parses
        the dx vector.
        """
        new = self.copy()
        for i, state in enumerate(new.value):
            new.value[i] = state.plus(dx[: state.dof])
            dx = dx[state.dof :]

        if new_stamp is not None:
            new.set_stamp_for_all(new_stamp)

        return new

    def minus(self, x: "CompositeState") -> np.ndarray:
        dx = []
        for i, v in enumerate(x.value):
            dx.append(
                self.value[i].minus(x.value[i]).reshape((self.value[i].dof,))
            )

        return np.concatenate(dx).reshape((-1, 1))

    def plus_by_id(
        self, dx, state_id: int, new_stamp: float = None
    ) -> "CompositeState":
        """
        Updates a specific sub-state.
        """
        new = self.copy()
        idx = new.get_index_by_id(state_id)
        new.value[idx].plus(dx)
        if new_stamp is not None:
            new.set_stamp_by_id(new_stamp, state_id)

        return new

    def jacobian_from_blocks(self, block_dict: dict):
        """
        Returns the jacobian of the entire composite state given jacobians
        associated with some of the substates. These are provided as a dictionary
        with the the keys being the substate IDs.
        """
        block: np.ndarray = list(block_dict.values())[0]
        m = block.shape[0]  # Dimension of "y" value
        jac = np.zeros((m, self.dof))
        slices = self.get_slices()
        for state_id, block in block_dict.items():
            slc = self.get_slice_by_id(state_id, slices)
            jac[:, slc] = block

        return jac

    def plus_jacobian(self, dx: np.ndarray) -> np.ndarray:
        dof = self.dof
        jac = np.zeros((dof, dof))
        counter = 0
        for state in self.value:
            jac[
                counter : counter + state.dof,
                counter : counter + state.dof,
            ] = state.plus_jacobian(dx[: state.dof])
            dx = dx[state.dof :]
            counter += state.dof

        return jac

    def minus_jacobian(self, x: "CompositeState") -> np.ndarray:
        dof = self.dof
        jac = np.zeros((dof, dof))
        counter = 0
        for i, state in enumerate(self.value):
            jac[
                counter : counter + state.dof,
                counter : counter + state.dof,
            ] = state.minus_jacobian(x.value[i])
            counter += state.dof

        return jac

    def __repr__(self):
        substate_line_list = []
        for v in self.value:
            substate_line_list.extend(v.__repr__().split("\n"))
        substates_str = "\n".join(["    " + s for s in substate_line_list])
        s = [
            f"{self.__class__.__name__}(stamp={self.stamp}, state_id={self.state_id}) with substates:",
            substates_str,
        ]
        return "\n".join(s)


class CompositeInput(Input):
    """
    <under development>
    """

    # TODO: add tests to new methods
    def __init__(self, input_list: List[Input]) -> None:
        self.input_list = input_list

    @property
    def dof(self) -> int:
        return sum([input.dof for input in self.input_list])

    @property
    def stamp(self) -> float:
        return self.input_list[0].stamp

    def get_index_by_id(self, state_id):
        """
        Get index of a particular state_id in the list of inputs.
        """
        return [x.state_id for x in self.input_list].index(state_id)

    def add_input(self, input: Input):
        """
        Adds an input to the composite input.
        """
        self.input_list.append(input)

    def remove_input_by_id(self, state_id):
        """
        Removes a given input by ID.
        """
        idx = self.get_index_by_id(state_id)
        self.input_list.pop(idx)

    def get_input_by_id(self, state_id) -> Input:
        """
        Get input object by id.
        """
        idx = self.get_index_by_id(state_id)
        return self.input_list[idx]

    def get_dof_by_id(self, state_id) -> int:
        """
        Get degrees of freedom of sub-input by id.
        """
        idx = self.get_index_by_id(state_id)
        return self.input_list[idx].dof

    def get_stamp_by_id(self, state_id) -> float:
        """
        Get timestamp of sub-input by id.
        """
        idx = self.get_index_by_id(state_id)
        return self.input_list[idx].stamp

    def set_stamp_by_id(self, stamp: float, state_id):
        """
        Set the timestamp of a sub-input by id.
        """
        idx = self.get_index_by_id(state_id)
        self.input_list[idx].stamp = stamp

    def set_input_by_id(self, input: Input, state_id):
        """
        Set the whole sub-input by id.
        """
        idx = self.get_index_by_id(state_id)
        self.input_list[idx] = input

    def set_stamp_for_all(self, stamp: float):
        """
        Set the timestamp of all subinputs.
        """
        for input in self.input_list:
            input.stamp = stamp

    def to_list(self):
        """
        Converts the CompositeInput object back into a list of inputs.
        """
        return self.input_list

    def copy(self) -> "CompositeInput":
        return CompositeInput([input.copy() for input in self.input_list])

    def plus(self, w: np.ndarray):
        new = self.copy()
        temp = w
        for i, input in enumerate(self.input_list):
            new.input_list[i] = input.plus(temp[: input.dof])
            temp = temp[input.dof :]

        return new


class CompositeProcessModel(ProcessModel):
    """
    <under development>
    Should this be called a StackedProcessModel?
    # TODO: Add documentation and tests
    """

    # TODO: This needs to be expanded and/or changed. We have come across the
    # following use cases:
    # 1. Applying a completely seperate process model to each sub-state.
    # 2. Applying a single process model to each sub-state (seperately).
    # 3. Applying a single process model to one sub-state, and leaving the rest
    #    unchanged.
    # 4. Applying process model A to some sub-states, and process model B to
    #    other sub-states
    # 5. Receiving a CompositeInput, which is a list of synchronously-received
    #    inputs, and applying each input to the corresponding sub-state.
    # 6. Receiving the state-specific input asynchronously, applying to the
    #    corresponding sub-state, and leaving the rest unchanged. Typically happens
    #    with case 3.

    # What they all have in common: list of decoupled process models, one per
    # substate. For coupled process models, the user will have to define their
    # own process model from scratch.

    def __init__(
        self,
        model_list: List[ProcessModel],
        shared_input: bool = False,
    ):
        self._model_list = model_list
        self._shared_input = shared_input

    def evaluate(
        self,
        x: CompositeState,
        u: CompositeInput,
        dt: float,
    ) -> CompositeState:
        x = x.copy()
        for i, x_sub in enumerate(x.value):
            if self._shared_input:
                u_sub = u
            else:
                u_sub = u.input_list[i]
            x.value[i] = self._model_list[i].evaluate(x_sub, u_sub, dt)

        return x

    def jacobian(
        self,
        x: CompositeState,
        u: CompositeInput,
        dt: float,
    ) -> np.ndarray:
        jac = []
        for i, x_sub in enumerate(x.value):
            if self._shared_input:
                u_sub = u
            else:
                u_sub = u.input_list[i]
            jac.append(self._model_list[i].jacobian(x_sub, u_sub, dt))

        return block_diag(*jac)

    def covariance(
        self,
        x: CompositeState,
        u: CompositeInput,
        dt: float,
    ) -> np.ndarray:
        cov = []
        for i, x_sub in enumerate(x.value):
            if self._shared_input:
                u_sub = u
            else:
                u_sub = u.input_list[i]
            cov.append(self._model_list[i].covariance(x_sub, u_sub, dt))

        return block_diag(*cov)


class CompositeMeasurementModel(MeasurementModel):
    """
    Wrapper for a standard measurement model that assigns the model to a specific
    substate (referenced by `state_id`) inside a CompositeState. This class
    will take care of extracting the relevant substate from the CompositeState,
    and then applying the measurement model to it. It will also take care of
    padding the Jacobian with zeros appropriately to match the degrees of freedom
    of the larger CompositeState.
    """

    def __init__(self, model: MeasurementModel, state_id):
        """
        Parameters
        ----------
        model : MeasurementModel
            Standard measurement model, which is appropriate only for a single
            substate in the CompositeState.
        state_id : Any
            The unique ID of the relevant substate in the CompositeState, to 
            assign the measurement model to.
        """
        self.model = model
        self.state_id = state_id

    def __repr__(self):
        return f"{self.model} (of substate '{self.state_id}')"

    def evaluate(self, x: CompositeState) -> np.ndarray:
        return self.model.evaluate(x.get_state_by_id(self.state_id))

    def jacobian(self, x: CompositeState) -> np.ndarray:
        x_sub = x.get_state_by_id(self.state_id)
        jac_sub = self.model.jacobian(x_sub)
        jac = np.zeros((jac_sub.shape[0], x.dof))
        slc = x.get_slice_by_id(self.state_id)
        jac[:, slc] = jac_sub
        return jac

    def covariance(self, x: CompositeState) -> np.ndarray:
        x_sub = x.get_state_by_id(self.state_id)
        return self.model.covariance(x_sub)


class CompositeMeasurement(Measurement):
    def __init__(self, y: Measurement, state_id: Any):
        """
        Converts a standard Measurement into a CompositeMeasurement, which
        replaces the model with a CompositeMeasurementModel.

        Parameters
        ----------
        y : Measurement
            Measurement to be converted.
        state_id : Any
            ID of the state that the measurement will be assigned to,
            as per the CompositeMeasurementModel.
        """
        model = CompositeMeasurementModel(y.model, state_id)
        super().__init__(
            value=y.value, stamp=y.stamp, model=model, state_id=y.state_id
        )
