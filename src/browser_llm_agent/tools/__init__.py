# Import all tool modules so @tool decorators execute and populate the registry.
import browser_llm_agent.tools.bash_tools      # noqa: F401
import browser_llm_agent.tools.file_tools      # noqa: F401
import browser_llm_agent.tools.workspace_tools # noqa: F401
import browser_llm_agent.tools.search_tools    # noqa: F401
import browser_llm_agent.tools.todo_tools      # noqa: F401
import browser_llm_agent.tools.memory_tools    # noqa: F401
import browser_llm_agent.tools.git_tools       # noqa: F401
import browser_llm_agent.tools.project_tools   # noqa: F401
import browser_llm_agent.tools.repl_tools      # noqa: F401
import browser_llm_agent.tools.browser_tools   # noqa: F401
import browser_llm_agent.tools.agent_tools     # noqa: F401
