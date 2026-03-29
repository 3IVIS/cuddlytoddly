
import git
import os
from pathlib import Path
import re
import shutil
import hashlib

from collections import deque, defaultdict

from cuddlytoddly.infra.logging import get_logger

logger = get_logger(__name__)

REPO_PATH = "dag_repo"
repo = None

node_to_commit = {}


def truncate_label(label, node_id=None, max_len=20):
    if node_id is not None:
        suffix = "#" + hashlib.sha256(node_id.encode()).hexdigest()[:6]
        return suffix
    else:
        suffix = "#" + hashlib.sha256(label.encode()).hexdigest()[:6]
        return suffix

def init_repo(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return git.Repo.init(path)



def graph_to_dag(snapshot):
    dag = {node_id: [] for node_id in snapshot}  # preserves insertion order
    for node_id, node in snapshot.items():
        for dep in node.dependencies:
            if dep in dag:
                dag[dep].append(node_id)  # children added in snapshot insertion order
    return dag

def topological_sort(dag):
    indegree = defaultdict(int)
    for node, children in dag.items():
        for child in children:
            indegree[child] += 1

    queue = deque([n for n in dag if indegree[n] == 0])  # insertion order, no sort
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for child in dag[node]:  # insertion order, no sort
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    return order

def commit_nodes_from_graph(snapshot):
    global node_to_commit
    node_to_commit.clear()

    repo_dir = Path(REPO_PATH)
    dag = graph_to_dag(snapshot)
    order = topological_sort(dag)

    # Detach HEAD so index.commit() doesn't chain via HEAD
    try:
        repo.head.reference = repo.commit("HEAD")
        repo.git.checkout("--detach", "HEAD")
    except Exception:
        pass  # Fine on empty repo

    for idx, node_id in enumerate(order):
        node = snapshot[node_id]
        parents = [
            node_to_commit[dep]
            for dep in sorted(node.dependencies)  # sort for deterministic commit parent ordering
            if dep in node_to_commit
        ]
        file_path = repo_dir / f"{node_id}.txt"
        file_path.write_text(f"This is node {node_id}\nStatus: {node.status}\n")
        repo.index.add([str(file_path.relative_to(REPO_PATH))])
        label = truncate_label(node.metadata.get("description") or node_id, node_id = node_id)
        try:
            parent_commits = [repo.commit(p) for p in parents]
            commit_obj = repo.index.commit(
                f"{label} [{node.status}]",
                parent_commits=parent_commits
            )
            node_to_commit[node_id] = commit_obj.hexsha  # use commit_obj, NOT repo.head
        except Exception as e:
            logger.exception("FAILED to commit node '%s': %s", node_id, e)

def compute_descendants(snapshot):
    reverse = defaultdict(set)
    for node_id, node in snapshot.items():
        for dep in node.dependencies:
            reverse[dep].add(node_id)

    descendants = defaultdict(set)

    def dfs(n, root):
        for child in reverse[n]:
            if child not in descendants[root]:
                descendants[root].add(child)
                dfs(child, root)

    for node_id in snapshot:
        dfs(node_id, node_id)

    return descendants

def commit_node_incremental(node_id, node, snapshot):
    """
    Incrementally commit a node. 
    Crucially, it uses the LATEST parent hashes from node_to_commit.
    """
    repo_dir = Path(REPO_PATH)
    file_path = repo_dir / f"{node_id}.txt"

    # 1. Update the physical file content
    file_path.write_text(
        f"This is node {node_id}\n"
        f"Dependencies: {list(node.dependencies)}\n"
        f"Status: {node.status}\n"
    )
    repo.index.add([str(file_path.relative_to(REPO_PATH))])

    # 2. Get the current Git hashes for parents from our tracking map
    # We use node_to_commit.get because a parent might not have a commit yet 
    # (though topological sort usually prevents this).
    parents = [
        node_to_commit[dep] for dep in sorted(node.dependencies) if dep in node_to_commit  # sort for determinism
    ]
    
    # 3. Create the commit
    try:
        # Convert hex strings to actual Git Commit objects
        parent_commits = [repo.commit(p) for p in parents] if parents else []
        label = truncate_label(node.metadata.get("description") or node_id, node_id = node_id)
        
        # In Git, changing parents creates a new hash. index.commit does this for us.
        commit_obj = repo.index.commit(
            f"{label} [{node.status}]", 
            parent_commits=parent_commits
        )
        
        # 4. Update the global map and node metadata
        node_to_commit[node_id] = commit_obj.hexsha
        node.metadata["last_commit_status"] = node.status
        node.metadata["last_commit_parents"] = sorted(parents)
        
        return True # Success
    except Exception as e:
        logger.exception("Failed to commit node '%s': %s", node_id, e)
        return False

def get_leaf_node_ids(dag):
    """Nodes with no children — tips of the DAG."""
    return {node_id for node_id, children in dag.items() if not children}

def sanitize_branch_name(node_id):
    """Replace characters invalid in Git branch names."""
    return re.sub(r'[^a-zA-Z0-9._-]', '_', node_id)

def update_tip_branches(snapshot):
    dag = graph_to_dag(snapshot)

    # Remove stale tip branches — skip any that fail (race condition)
    for branch in list(repo.heads):
        if branch.name.startswith("tip_"):
            try:
                repo.delete_head(branch, force=True)
            except Exception as e:
                logger.debug("[GIT] Could not delete branch %s: %s", branch.name, e)

    # Create a tip for every node that has a commit
    for node_id in snapshot:
        if node_id in node_to_commit:
            branch_name = f"tip_{sanitize_branch_name(node_id)}"
            try:
                repo.create_head(branch_name, node_to_commit[node_id], force=True)
            except Exception as e:
                logger.debug("[GIT] Could not create branch %s: %s", branch_name, e)

    # Point master at root
    root_id = next(
        (nid for nid in snapshot
         if not snapshot[nid].dependencies and nid in node_to_commit),
        None
    )
    if root_id:
        try:
            repo.create_head("master", node_to_commit[root_id], force=True)
        except Exception as e:
            logger.debug("[GIT] Could not update master: %s", e)
            
def rebuild_repo_from_graph(graph, incremental=True):
    try:
        global repo
        snapshot = graph.get_snapshot()
        # Exclude hidden execution step nodes from git projection
        snapshot = {
            nid: n for nid, n in snapshot.items()
            if not (n.node_type == "execution_step"
                    and n.metadata.get("hidden", False))
        }
        dag = graph_to_dag(snapshot)

        try:
            if repo is None:
                raise ValueError("no repo yet")
            repo.head.commit
        except (ValueError, TypeError):
            incremental = False

        if not incremental:
            repo = init_repo(REPO_PATH)
            node_to_commit.clear()
            commit_nodes_from_graph(snapshot)
            update_tip_branches(snapshot)
            return

        # Topological sort of DAG nodes
        order = topological_sort(dag)
        dirty = set()

        # 1️⃣ Mark nodes as dirty if status changed, parents changed, or missing
        for node_id in order:
            node = snapshot[node_id]
            last_status = node.metadata.get("last_commit_status")
            last_parents = node.metadata.get("last_commit_parents", [])
            missing_parent = False
            resolved_parents = []

            for dep in node.dependencies:
                if dep not in node_to_commit:
                    missing_parent = True
                else:
                    resolved_parents.append(node_to_commit[dep])

            current_parents = sorted(resolved_parents)

            if missing_parent:
                dirty.add(node_id)
            if node_id not in node_to_commit or node.status != last_status or current_parents != last_parents:
                dirty.add(node_id)

        # 2️⃣ Propagate dirty downward (dependencies)
        for node_id in order:
            if any(dep in dirty for dep in snapshot[node_id].dependencies):
                dirty.add(node_id)

        # 3️⃣ Propagate dirty upward (children)
        changed = True
        while changed:
            changed = False
            for node_id in order:
                if node_id in dirty:
                    for child_id in snapshot[node_id].children:
                        if child_id not in dirty and child_id in snapshot:
                            dirty.add(child_id)
                            changed = True

        # 4️⃣ Ensure **all snapshot nodes** are included (orphans, missing in DAG)
        full_order = order + [n for n in snapshot if n not in order]
        for node_id in snapshot:
            if node_id not in node_to_commit:
                dirty.add(node_id)

        # 5️⃣ Detach HEAD safely (pick any committed SHA or fallback)
        if node_to_commit:
            some_sha = next(iter(node_to_commit.values()))
            repo.git.checkout(some_sha)
        else:
            # fallback: detached HEAD at empty tree
            repo.git.checkout("--orphan", "tmp_head")
            repo.index.reset()

        # 6️⃣ Commit all dirty nodes
        for node_id in full_order:
            if node_id in dirty or node_id not in node_to_commit:
                commit_node_incremental(node_id, snapshot[node_id], snapshot)

        # 7️⃣ Reset index and update tip branches
        repo.index.reset()
        update_tip_branches(snapshot)

        # 8️⃣ Optional Git garbage collection
        try:
            repo.git.gc(prune="now")
        except Exception as e:
            logger.warning("gc warning (non-fatal): %s", e)
    except Exception as e:
        logger.error("[GIT] rebuild_repo_from_graph failed: %s", e, exc_info=True)
        # Last resort: nuke and rebuild from scratch
        try:
            repo = init_repo(REPO_PATH)
            node_to_commit.clear()
            commit_nodes_from_graph(snapshot)
            update_tip_branches(snapshot)
            logger.info("[GIT] Full rebuild succeeded after error")
        except Exception as e2:
            logger.error("[GIT] Full rebuild also failed: %s", e2)

def delete_node(node_id, graph):
    """
    Soft-delete a node:
    - Mark as deleted
    - Commit the deletion
    """
    if node_id in graph.nodes:
        node = graph.nodes[node_id]
        node.metadata["deleted"] = True
        commit_node_incremental(node_id, node, graph.get_snapshot())