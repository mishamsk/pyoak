from __future__ import annotations

import uuid
import weakref
from random import Random
from typing import TYPE_CHECKING, Callable, Sequence

if TYPE_CHECKING:
    from .node import ASTNode


_REF_TO_NODE: weakref.WeakValueDictionary[str, ASTNode] = weakref.WeakValueDictionary()
"""Registry of all refs to node objects."""


def _get_unique_ref() -> str:
    """Pyoak private internal API to generate a new ref value.

    Returns:
        str: The new ref value.
    """
    return uuid.uuid4().hex


def _get_test_ref_generator(rd: Random) -> Callable[[], str]:
    """Pyoak private internal API to generate a new ref value in a reproducible
    manner for testing.

    Returns:
        str: The new ref value.
    """

    def _inner() -> str:
        """Pyoak private internal API to generate a new ref value.

        Returns:
            str: The new ref value.
        """
        return uuid.UUID(int=rd.getrandbits(128), version=4).hex

    return _inner


_ID_GEN = _get_unique_ref
"""The ref ID generator to use."""


def _register(node: ASTNode) -> str:
    """Pyoak private internal API to add a node to the registry.

    Args:
        node (ASTNode): The node to add.

    Raises:
        RuntimeError: In an unlikely event of UUID collision.

    Returns:
        str: The ref value of the node.
    """

    # Find a free ref value
    ref = _ID_GEN()
    if ref in _REF_TO_NODE:
        raise RuntimeError("Failed to find a free ref value")

    _REF_TO_NODE[ref] = node

    return ref


def _register_with_ref(node: ASTNode, ref: str) -> bool:
    """Pyoak private internal API to add a node to the registry using a known
    ref (only used for deserializtion).

    Args:
        node (ASTNode): The node to add.
        ref (str | None, optional): The exected ref value to use.


    Returns:
        bool: True if the node was added, False if the ref was already in use.
    """

    if ref in _REF_TO_NODE:
        return False

    _REF_TO_NODE[ref] = node

    return True


_get_node = _REF_TO_NODE.get


def _pop_node(node: ASTNode) -> ASTNode | None:
    """Pyoak private internal API to remove a node from the registry.

    Args:
        node (ASTNode): The node to remove.

    Returns:
        ASTNode | None: The removed node or None if the node was not in the registry.
    """
    if node.ref is None:
        return None

    return _REF_TO_NODE.pop(node.ref, None)


def set_seed(seed: int) -> None:
    """Set the seed of the random ref ID generator.

    Use for testing purposes only.

    Args:
        seed (int): The seed to set.
    """
    global _ID_GEN

    _ID_GEN = _get_test_ref_generator(Random(seed))


def reset_seed() -> None:
    """Reset the seed of the random ID generator.

    Use for testing purposes only.
    """
    global _ID_GEN

    _ID_GEN = _get_unique_ref


def clear_registry() -> None:
    """Clear the registry.

    Important: this function doesn't remove ref values from the nodes,
    and thus breaks some of the impicit contracts for node-registry interaction.

    Use for testing purposes only.
    """
    _REF_TO_NODE.clear()


def pop_nodes(nodes: Sequence[ASTNode]) -> None:
    """Removes the nodes from the registry.

    Important: this function doesn't remove ref values from the nodes,
    and thus breaks some of the impicit contracts for node-registry interaction.

    Use for testing purposes only.

    Args:
        nodes (Sequence[ASTNode]): The nodes to remove.
    """
    for node in filter(lambda n: n.ref is not None, nodes):
        _REF_TO_NODE.pop(node.ref, None)  # type: ignore[arg-type]
