from abc import ABC, abstractmethod


class AbstractRoute(ABC):
    """
    Inherit this class for xroute to route
    """

    @abstractmethod
    def __init__(self, seed=None):
        pass

    @abstractmethod
    def step(self, action):
        """
        Apply action to route.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the route has ended.
        """
        pass

    def step_inference(self, action_list):
        """
        Apply action to route.

        Args:
            action_list : actions of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the route has ended.
        """
        pass

    @abstractmethod
    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the route have to be able to handle one of returned actions.

        For complex routing where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the route for a new route.

        Returns:
            Initial observation of the route.
        """
        pass

    def reset_inference(self):
        """
        Reset the route for a new route inference.

        Returns:
            Initial observation of the route.
        """
        pass

    def close(self):
        """
        Properly close the route.
        """
        pass

    @abstractmethod
    def render(self):
        """
        Display the route observation.
        """
        pass

    def human_to_action(self):
        """
        For route testing, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        choice = input(f"Enter the action to route (legal actions {self.legal_actions()}): ")
        while int(choice) not in self.legal_actions():
            choice = input("Illegal action. Enter another action : ")
        return int(choice)

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        return str(action_number)
