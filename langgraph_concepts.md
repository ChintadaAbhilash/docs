Skip to content

To learn more about LangGraph, check out our first LangChain Academy course,
_Introduction to LangGraph_ , available for free here.

![logo](./Agent architectures_files/wordmark_dark.svg) ![logo](./Agent
architectures_files/wordmark_light.svg)

Agent architectures

Type to start searching

GitHub

  * 0.2.62
  * 7.9k
  * 1.3k

  * Home 
  * Tutorials 
  * How-to Guides 
  * Conceptual Guides 
  * Reference 

![logo](./Agent architectures_files/wordmark_dark.svg) ![logo](./Agent
architectures_files/wordmark_light.svg)

GitHub

  * 0.2.62
  * 7.9k
  * 1.3k

  * Home 
  * Tutorials 
  * How-to Guides 
  * Conceptual Guides 

Conceptual Guides

    * LangGraph  LangGraph 
      * LangGraph 
      * Why LangGraph? 
      * LangGraph Glossary 
      * Agent architectures  Agent architectures  Table of contents 
        * Router 
          * Structured Output 
        * Tool calling agent 
          * Tool calling 
          * Memory 
          * Planning 
          * ReAct implementation 
        * Custom agent architectures 
          * Human-in-the-loop 
          * Parallelization 
          * Subgraphs 
          * Reflection 
      * Multi-agent Systems 
      * Human-in-the-loop 
      * Persistence 
      * Memory 
      * Streaming 
      * FAQ 
    * LangGraph Platform  LangGraph Platform 
      * LangGraph Platform 
      * High Level 
      * Components 
      * LangGraph Server 
      * Deployment Options 
  * Reference 

Table of contents

  * Router 
    * Structured Output 
  * Tool calling agent 
    * Tool calling 
    * Memory 
    * Planning 
    * ReAct implementation 
  * Custom agent architectures 
    * Human-in-the-loop 
    * Parallelization 
    * Subgraphs 
    * Reflection 

  1. Home 
  2. Conceptual Guides 
  3. LangGraph 

# Agent architectures¶

Many LLM applications implement a particular control flow of steps before and
/ or after LLM calls. As an example, RAG performs retrieval of documents
relevant to a user question, and passes those documents to an LLM in order to
ground the model's response in the provided document context.

Instead of hard-coding a fixed control flow, we sometimes want LLM systems
that can pick their own control flow to solve more complex problems! This is
one definition of an agent: _an agent is a system that uses an LLM to decide
the control flow of an application._ There are many ways that an LLM can
control application:

  * An LLM can route between two potential paths
  * An LLM can decide which of many tools to call
  * An LLM can decide whether the generated answer is sufficient or more work is needed

As a result, there are many different types of agent architectures, which give
an LLM varying levels of control.

![Agent Types](./Agent architectures_files/agent_types.png)

## Router¶

A router allows an LLM to select a single step from a specified set of
options. This is an agent architecture that exhibits a relatively limited
level of control because the LLM usually focuses on making a single decision
and produces a specific output from limited set of pre-defined options.
Routers typically employ a few different concepts to achieve this.

### Structured Output¶

Structured outputs with LLMs work by providing a specific format or schema
that the LLM should follow in its response. This is similar to tool calling,
but more general. While tool calling typically involves selecting and using
predefined functions, structured outputs can be used for any type of formatted
response. Common methods to achieve structured outputs include:

  1. Prompt engineering: Instructing the LLM to respond in a specific format via the system prompt.
  2. Output parsers: Using post-processing to extract structured data from LLM responses.
  3. Tool calling: Leveraging built-in tool calling capabilities of some LLMs to generate structured outputs.

Structured outputs are crucial for routing as they ensure the LLM's decision
can be reliably interpreted and acted upon by the system. Learn more about
structured outputs in this how-to guide.

## Tool calling agent¶

While a router allows an LLM to make a single decision, more complex agent
architectures expand the LLM's control in two key ways:

  1. Multi-step decision making: The LLM can make a series of decisions, one after another, instead of just one.
  2. Tool access: The LLM can choose from and use a variety of tools to accomplish tasks.

ReAct is a popular general purpose agent architecture that combines these
expansions, integrating three core concepts.

  1. `Tool calling`: Allowing the LLM to select and use various tools as needed.
  2. `Memory`: Enabling the agent to retain and use information from previous steps.
  3. `Planning`: Empowering the LLM to create and follow multi-step plans to achieve goals.

This architecture allows for more complex and flexible agent behaviors, going
beyond simple routing to enable dynamic problem-solving with multiple steps.
You can use it with `create_react_agent`.

### Tool calling¶

Tools are useful whenever you want an agent to interact with external systems.
External systems (e.g., APIs) often require a particular input schema or
payload, rather than natural language. When we bind an API, for example, as a
tool, we give the model awareness of the required input schema. The model will
choose to call a tool based upon the natural language input from the user and
it will return an output that adheres to the tool's required schema.

Many LLM providers support tool calling and tool calling interface in
LangChain is simple: you can simply pass any Python `function` into
`ChatModel.bind_tools(function)`.

![Tools](./Agent architectures_files/tool_call.png)

### Memory¶

Memory is crucial for agents, enabling them to retain and utilize information
across multiple steps of problem-solving. It operates on different scales:

  1. Short-term memory: Allows the agent to access information acquired during earlier steps in a sequence.
  2. Long-term memory: Enables the agent to recall information from previous interactions, such as past messages in a conversation.

LangGraph provides full control over memory implementation:

  * `State`: User-defined schema specifying the exact structure of memory to retain.
  * `Checkpointers`: Mechanism to store state at every step across different interactions.

This flexible approach allows you to tailor the memory system to your specific
agent architecture needs. For a practical guide on adding memory to your
graph, see this tutorial.

Effective memory management enhances an agent's ability to maintain context,
learn from past experiences, and make more informed decisions over time.

### Planning¶

In the ReAct architecture, an LLM is called repeatedly in a while-loop. At
each step the agent decides which tools to call, and what the inputs to those
tools should be. Those tools are then executed, and the outputs are fed back
into the LLM as observations. The while-loop terminates when the agent decides
it has enough information to solve the user request and it is not worth
calling any more tools.

### ReAct implementation¶

There are several differences between this paper and the pre-built
`create_react_agent` implementation:

  * First, we use tool-calling to have LLMs call tools, whereas the paper used prompting + parsing of raw output. This is because tool calling did not exist when the paper was written, but is generally better and more reliable.
  * Second, we use messages to prompt the LLM, whereas the paper used string formatting. This is because at the time of writing, LLMs didn't even expose a message-based interface, whereas now that's the only interface they expose.
  * Third, the paper required all inputs to the tools to be a single string. This was largely due to LLMs not being super capable at the time, and only really being able to generate a single input. Our implementation allows for using tools that require multiple inputs.
  * Fourth, the paper only looks at calling a single tool at the time, largely due to limitations in LLMs performance at the time. Our implementation allows for calling multiple tools at a time.
  * Finally, the paper asked the LLM to explicitly generate a "Thought" step before deciding which tools to call. This is the "Reasoning" part of "ReAct". Our implementation does not do this by default, largely because LLMs have gotten much better and that is not as necessary. Of course, if you wish to prompt it do so, you certainly can.

## Custom agent architectures¶

While routers and tool-calling agents (like ReAct) are common, customizing
agent architectures often leads to better performance for specific tasks.
LangGraph offers several powerful features for building tailored agent
systems:

### Human-in-the-loop¶

Human involvement can significantly enhance agent reliability, especially for
sensitive tasks. This can involve:

  * Approving specific actions
  * Providing feedback to update the agent's state
  * Offering guidance in complex decision-making processes

Human-in-the-loop patterns are crucial when full automation isn't feasible or
desirable. Learn more in our human-in-the-loop guide.

### Parallelization¶

Parallel processing is vital for efficient multi-agent systems and complex
tasks. LangGraph supports parallelization through its Send API, enabling:

  * Concurrent processing of multiple states
  * Implementation of map-reduce-like operations
  * Efficient handling of independent subtasks

For practical implementation, see our map-reduce tutorial.

### Subgraphs¶

Subgraphs are essential for managing complex agent architectures, particularly
in multi-agent systems. They allow:

  * Isolated state management for individual agents
  * Hierarchical organization of agent teams
  * Controlled communication between agents and the main system

Subgraphs communicate with the parent graph through overlapping keys in the
state schema. This enables flexible, modular agent design. For implementation
details, refer to our subgraph how-to guide.

### Reflection¶

Reflection mechanisms can significantly improve agent reliability by:

  1. Evaluating task completion and correctness
  2. Providing feedback for iterative improvement
  3. Enabling self-correction and learning

While often LLM-based, reflection can also use deterministic methods. For
instance, in coding tasks, compilation errors can serve as feedback. This
approach is demonstrated in this video using LangGraph for self-corrective
code generation.

By leveraging these features, LangGraph enables the creation of sophisticated,
task-specific agent architectures that can handle complex workflows,
collaborate effectively, and continuously improve their performance.

## Comments

Back to top

Previous

LangGraph Glossary

Next

Multi-agent Systems

Made with  Material for MkDocs Insiders



Skip to content

To learn more about LangGraph, check out our first LangChain Academy course,
_Introduction to LangGraph_ , available for free here.

![logo](./Breakpoints_files/wordmark_dark.svg)
![logo](./Breakpoints_files/wordmark_light.svg)

Breakpoints

Type to start searching

GitHub

  * 0.2.62
  * 7.9k
  * 1.3k

  * Home 
  * Tutorials 
  * How-to Guides 
  * Conceptual Guides 
  * Reference 

![logo](./Breakpoints_files/wordmark_dark.svg)
![logo](./Breakpoints_files/wordmark_light.svg)

GitHub

  * 0.2.62
  * 7.9k
  * 1.3k

  * Home 
  * Tutorials 
  * How-to Guides 
  * Conceptual Guides 
  * Reference 

Table of contents

  * Requirements 
  * Setting breakpoints 
    * Static breakpoints 
    * NodeInterrupt exception 
  * Additional Resources 📚 

# Breakpoints¶

Breakpoints pause graph execution at specific points and enable stepping
through execution step by step. Breakpoints are powered by LangGraph's
**persistence layer** , which saves the state after each graph step.
Breakpoints can also be used to enable **human-in-the-loop** workflows, though
we recommend using the `interrupt` function for this purpose.

## Requirements¶

To use breakpoints, you will need to:

  1. **Specify a checkpointer** to save the graph state after each step.
  2. **Set breakpoints** to specify where execution should pause.
  3. **Run the graph** with a **thread ID** to pause execution at the breakpoint.
  4. **Resume execution** using `invoke`/`ainvoke`/`stream`/`astream` (see **The`Command` primitive**).

## Setting breakpoints¶

There are two places where you can set breakpoints:

  1. **Before** or **after** a node executes by setting breakpoints at **compile time** or **run time**. We call these **static breakpoints**.
  2. **Inside** a node using the `NodeInterrupt` exception.

### Static breakpoints¶

Static breakpoints are triggered either **before** or **after** a node
executes. You can set static breakpoints by specifying `interrupt_before` and
`interrupt_after` at **"compile" time** or **run time**.

Compile timeRun time

    
    
    graph = graph_builder.compile(
        interrupt_before=["node_a"], 
        interrupt_after=["node_b", "node_c"],
        checkpointer=..., # Specify a checkpointer
    )
    
    thread_config = {
        "configurable": {
            "thread_id": "some_thread"
        }
    }
    
    # Run the graph until the breakpoint
    graph.invoke(inputs, config=thread_config)
    
    # Optionally update the graph state based on user input
    graph.update_state(update, config=thread_config)
    
    # Resume the graph
    graph.invoke(None, config=thread_config)
    
    
    
    graph.invoke(
        inputs, 
        config={"configurable": {"thread_id": "some_thread"}}, 
        interrupt_before=["node_a"], 
        interrupt_after=["node_b", "node_c"]
    )
    
    thread_config = {
        "configurable": {
            "thread_id": "some_thread"
        }
    }
    
    # Run the graph until the breakpoint
    graph.invoke(inputs, config=thread_config)
    
    # Optionally update the graph state based on user input
    graph.update_state(update, config=thread_config)
    
    # Resume the graph
    graph.invoke(None, config=thread_config)
    

Note

You cannot set static breakpoints at runtime for **sub-graphs**. If you have a
sub-graph, you must set the breakpoints at compilation time.

Static breakpoints can be especially useful for debugging if you want to step
through the graph execution one node at a time or if you want to pause the
graph execution at specific nodes.

### `NodeInterrupt` exception¶

We recommend that you **use the`interrupt` function instead** of the
`NodeInterrupt` exception if you're trying to implement human-in-the-loop
workflows. The `interrupt` function is easier to use and more flexible.

`NodeInterrupt` exception

The developer can define some _condition_ that must be met for a breakpoint to
be triggered. This concept of dynamic breakpoints is useful when the developer
wants to halt the graph under _a particular condition_. This uses a
`NodeInterrupt`, which is a special type of exception that can be raised from
within a node based upon some condition. As an example, we can define a
dynamic breakpoint that triggers when the `input` is longer than 5 characters.

    
    
    def my_node(state: State) -> State:
        if len(state['input']) > 5:
            raise NodeInterrupt(f"Received input that is longer than 5 characters: {state['input']}")
    
        return state
    

Let's assume we run the graph with an input that triggers the dynamic
breakpoint and then attempt to resume the graph execution simply by passing in
`None` for the input.

    
    
    # Attempt to continue the graph execution with no change to state after we hit the dynamic breakpoint 
    for event in graph.stream(None, thread_config, stream_mode="values"):
        print(event)
    

The graph will _interrupt_ again because this node will be _re-run_ with the
same graph state. We need to change the graph state such that the condition
that triggers the dynamic breakpoint is no longer met. So, we can simply edit
the graph state to an input that meets the condition of our dynamic breakpoint
(< 5 characters) and re-run the node.

    
    
    # Update the state to pass the dynamic breakpoint
    graph.update_state(config=thread_config, values={"input": "foo"})
    for event in graph.stream(None, thread_config, stream_mode="values"):
        print(event)
    

Alternatively, what if we want to keep our current input and skip the node
(`my_node`) that performs the check? To do this, we can simply perform the
graph update with `as_node="my_node"` and pass in `None` for the values. This
will make no update the graph state, but run the update as `my_node`,
effectively skipping the node and bypassing the dynamic breakpoint.

    
    
    # This update will skip the node `my_node` altogether
    graph.update_state(config=thread_config, values=None, as_node="my_node")
    for event in graph.stream(None, thread_config, stream_mode="values"):
        print(event)
    

## Additional Resources 📚¶

  * **Conceptual Guide: Persistence** : Read the persistence guide for more context about persistence.
  * **Conceptual Guide: Human-in-the-loop** : Read the human-in-the-loop guide for more context on integrating human feedback into LangGraph applications using breakpoints.
  * **How to View and Update Past Graph State** : Step-by-step instructions for working with graph state that demonstrate the **replay** and **fork** actions.

## Comments

Back to top

Made with  Material for MkDocs Insiders



Skip to content

To learn more about LangGraph, check out our first LangChain Academy course,
_Introduction to LangGraph_ , available for free here.

![logo](./Human-in-the-loop_files/wordmark_dark.svg) ![logo](./Human-in-the-
loop_files/wordmark_light.svg)

Human-in-the-loop

Type to start searching

GitHub

  * 0.2.62
  * 7.9k
  * 1.3k

  * Home 
  * Tutorials 
  * How-to Guides 
  * Conceptual Guides 
  * Reference 

![logo](./Human-in-the-loop_files/wordmark_dark.svg) ![logo](./Human-in-the-
loop_files/wordmark_light.svg)

GitHub

  * 0.2.62
  * 7.9k
  * 1.3k

  * Home 
  * Tutorials 
  * How-to Guides 
  * Conceptual Guides 

Conceptual Guides

    * LangGraph  LangGraph 
      * LangGraph 
      * Why LangGraph? 
      * LangGraph Glossary 
      * Agent architectures 
      * Multi-agent Systems 
      * Human-in-the-loop  Human-in-the-loop  Table of contents 
        * Use cases 
        * interrupt 
        * Requirements 
        * Design Patterns 
          * Approve or Reject 
          * Review & Edit State 
          * Review Tool Calls 
          * Multi-turn conversation 
          * Validating human input 
        * The Command primitive 
        * Using with invoke and ainvoke 
        * How does resuming from an interrupt work? 
        * Common Pitfalls 
          * Side-effects 
          * Subgraphs called as functions 
          * Using multiple interrupts 
        * Additional Resources 📚 
      * Persistence 
      * Memory 
      * Streaming 
      * FAQ 
    * LangGraph Platform  LangGraph Platform 
      * LangGraph Platform 
      * High Level 
      * Components 
      * LangGraph Server 
      * Deployment Options 
  * Reference 

Table of contents

  * Use cases 
  * interrupt 
  * Requirements 
  * Design Patterns 
    * Approve or Reject 
    * Review & Edit State 
    * Review Tool Calls 
    * Multi-turn conversation 
    * Validating human input 
  * The Command primitive 
  * Using with invoke and ainvoke 
  * How does resuming from an interrupt work? 
  * Common Pitfalls 
    * Side-effects 
    * Subgraphs called as functions 
    * Using multiple interrupts 
  * Additional Resources 📚 

  1. Home 
  2. Conceptual Guides 
  3. LangGraph 

# Human-in-the-loop¶

This guide uses the new `interrupt` function.

As of LangGraph 0.2.57, the recommended way to set breakpoints is using the
`interrupt` function as it simplifies **human-in-the-loop** patterns.

If you're looking for the previous version of this conceptual guide, which
relied on static breakpoints and `NodeInterrupt` exception, it is available
here.

A **human-in-the-loop** (or "on-the-loop") workflow integrates human input
into automated processes, allowing for decisions, validation, or corrections
at key stages. This is especially useful in **LLM-based applications** , where
the underlying model may generate occasional inaccuracies. In low-error-
tolerance scenarios like compliance, decision-making, or content generation,
human involvement ensures reliability by enabling review, correction, or
override of model outputs.

## Use cases¶

Key use cases for **human-in-the-loop** workflows in LLM-based applications
include:

  1. **🛠️ Reviewing tool calls** : Humans can review, edit, or approve tool calls requested by the LLM before tool execution.
  2. **✅ Validating LLM outputs** : Humans can review, edit, or approve content generated by the LLM.
  3. **💡 Providing context** : Enable the LLM to explicitly request human input for clarification or additional details or to support multi-turn conversations.

## `interrupt`¶

The `interrupt` function in LangGraph enables human-in-the-loop workflows by
pausing the graph at a specific node, presenting information to a human, and
resuming the graph with their input. This function is useful for tasks like
approvals, edits, or collecting additional input. The `interrupt` function is
used in conjunction with the `Command` object to resume the graph with a value
provided by the human.

    
    
    from langgraph.types import interrupt
    
    def human_node(state: State):
        value = interrupt(
            # Any JSON serializable value to surface to the human.
            # For example, a question or a piece of text or a set of keys in the state
           {
              "text_to_revise": state["some_text"]
           }
        )
        # Update the state with the human's input or route the graph based on the input.
        return {
            "some_text": value
        }
    
    graph = graph_builder.compile(
        checkpointer=checkpointer # Required for `interrupt` to work
    )
    
    # Run the graph until the interrupt
    thread_config = {"configurable": {"thread_id": "some_id"}}
    graph.invoke(some_input, config=thread_config)
    
    # Resume the graph with the human's input
    graph.invoke(Command(resume=value_from_human), config=thread_config)
    

API Reference: interrupt

    
    
    {'some_text': 'Edited text'}
    

Warning

Interrupts are both powerful and ergonomic. However, while they may resemble
Python's input() function in terms of developer experience, it's important to
note that they do not automatically resume execution from the interruption
point. Instead, they rerun the entire node where the interrupt was used. For
this reason, interrupts are typically best placed at the start of a node or in
a dedicated node. Please read the resuming from an interrupt section for more
details.

Full Code

Here's a full example of how to use `interrupt` in a graph, if you'd like to
see the code in action.

    
    
    from typing import TypedDict
    import uuid
    
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.constants import START
    from langgraph.graph import StateGraph
    from langgraph.types import interrupt, Command
    
    class State(TypedDict):
       """The graph state."""
       some_text: str
    
    def human_node(state: State):
       value = interrupt(
          # Any JSON serializable value to surface to the human.
          # For example, a question or a piece of text or a set of keys in the state
          {
             "text_to_revise": state["some_text"]
          }
       )
       return {
          # Update the state with the human's input
          "some_text": value
       }
    
    
    # Build the graph
    graph_builder = StateGraph(State)
    # Add the human-node to the graph
    graph_builder.add_node("human_node", human_node)
    graph_builder.add_edge(START, "human_node")
    
    # A checkpointer is required for `interrupt` to work.
    checkpointer = MemorySaver()
    graph = graph_builder.compile(
       checkpointer=checkpointer
    )
    
    # Pass a thread ID to the graph to run it.
    thread_config = {"configurable": {"thread_id": uuid.uuid4()}}
    
    # Using stream() to directly surface the `__interrupt__` information.
    for chunk in graph.stream({"some_text": "Original text"}, config=thread_config):
       print(chunk)
    
    # Resume using Command
    for chunk in graph.stream(Command(resume="Edited text"), config=thread_config):
       print(chunk)
    

API Reference: MemorySaver | START | StateGraph | interrupt | Command
    
    
    {'__interrupt__': (
          Interrupt(
             value={'question': 'Please revise the text', 'some_text': 'Original text'}, 
             resumable=True, 
             ns=['human_node:10fe492f-3688-c8c6-0d0a-ec61a43fecd6'], 
             when='during'
          ),
       )
    }
    {'human_node': {'some_text': 'Edited text'}}
    

## Requirements¶

To use `interrupt` in your graph, you need to:

  1. **Specify a checkpointer** to save the graph state after each step.
  2. **Call`interrupt()`** in the appropriate place. See the Design Patterns section for examples.
  3. **Run the graph** with a **thread ID** until the `interrupt` is hit.
  4. **Resume execution** using `invoke`/`ainvoke`/`stream`/`astream` (see **The`Command` primitive**).

## Design Patterns¶

There are typically three different **actions** that you can do with a human-
in-the-loop workflow:

  1. **Approve or Reject** : Pause the graph before a critical step, such as an API call, to review and approve the action. If the action is rejected, you can prevent the graph from executing the step, and potentially take an alternative action. This pattern often involve **routing** the graph based on the human's input.
  2. **Edit Graph State** : Pause the graph to review and edit the graph state. This is useful for correcting mistakes or updating the state with additional information. This pattern often involves **updating** the state with the human's input.
  3. **Get Input** : Explicitly request human input at a particular step in the graph. This is useful for collecting additional information or context to inform the agent's decision-making process or for supporting **multi-turn conversations**.

Below we show different design patterns that can be implemented using these
**actions**.

### Approve or Reject¶

![image](./Human-in-the-loop_files/approve-or-reject.png)

Depending on the human's approval or rejection, the graph can proceed with the
action or take an alternative path.

Pause the graph before a critical step, such as an API call, to review and
approve the action. If the action is rejected, you can prevent the graph from
executing the step, and potentially take an alternative action.

    
    
    from typing import Literal
    from langgraph.types import interrupt, Command
    
    def human_approval(state: State) -> Command[Literal["some_node", "another_node"]]:
        is_approved = interrupt(
            {
                "question": "Is this correct?",
                # Surface the output that should be
                # reviewed and approved by the human.
                "llm_output": state["llm_output"]
            }
        )
    
        if is_approved:
            return Command(goto="some_node")
        else:
            return Command(goto="another_node")
    
    # Add the node to the graph in an appropriate location
    # and connect it to the relevant nodes.
    graph_builder.add_node("human_approval", human_approval)
    graph = graph_builder.compile(checkpointer=checkpointer)
    
    # After running the graph and hitting the interrupt, the graph will pause.
    # Resume it with either an approval or rejection.
    thread_config = {"configurable": {"thread_id": "some_id"}}
    graph.invoke(Command(resume=True), config=thread_config)
    

API Reference: interrupt | Command

See how to review tool calls for a more detailed example.

### Review & Edit State¶

![image](./Human-in-the-loop_files/edit-graph-state-simple.png)

A human can review and edit the state of the graph. This is useful for
correcting mistakes or updating the state with additional information.

    
    
    from langgraph.types import interrupt
    
    def human_editing(state: State):
        ...
        result = interrupt(
            # Interrupt information to surface to the client.
            # Can be any JSON serializable value.
            {
                "task": "Review the output from the LLM and make any necessary edits.",
                "llm_generated_summary": state["llm_generated_summary"]
            }
        )
    
        # Update the state with the edited text
        return {
            "llm_generated_summary": result["edited_text"] 
        }
    
    # Add the node to the graph in an appropriate location
    # and connect it to the relevant nodes.
    graph_builder.add_node("human_editing", human_editing)
    graph = graph_builder.compile(checkpointer=checkpointer)
    
    ...
    
    # After running the graph and hitting the interrupt, the graph will pause.
    # Resume it with the edited text.
    thread_config = {"configurable": {"thread_id": "some_id"}}
    graph.invoke(
        Command(resume={"edited_text": "The edited text"}), 
        config=thread_config
    )
    

API Reference: interrupt

See How to wait for user input using interrupt for a more detailed example.

### Review Tool Calls¶

![image](./Human-in-the-loop_files/tool-call-review.png)

A human can review and edit the output from the LLM before proceeding. This is
particularly critical in applications where the tool calls requested by the
LLM may be sensitive or require human oversight.

    
    
    def human_review_node(state) -> Command[Literal["call_llm", "run_tool"]]:
        # This is the value we'll be providing via Command(resume=<human_review>)
        human_review = interrupt(
            {
                "question": "Is this correct?",
                # Surface tool calls for review
                "tool_call": tool_call
            }
        )
    
        review_action, review_data = human_review
    
        # Approve the tool call and continue
        if review_action == "continue":
            return Command(goto="run_tool")
    
        # Modify the tool call manually and then continue
        elif review_action == "update":
            ...
            updated_msg = get_updated_msg(review_data)
            # Remember that to modify an existing message you will need
            # to pass the message with a matching ID.
            return Command(goto="run_tool", update={"messages": [updated_message]})
    
        # Give natural language feedback, and then pass that back to the agent
        elif review_action == "feedback":
            ...
            feedback_msg = get_feedback_msg(review_data)
            return Command(goto="call_llm", update={"messages": [feedback_msg]})
    

See how to review tool calls for a more detailed example.

### Multi-turn conversation¶

![image](./Human-in-the-loop_files/multi-turn-conversation.png)

A **multi-turn conversation** architecture where an **agent** and **human
node** cycle back and forth until the agent decides to hand off the
conversation to another agent or another part of the system.

A **multi-turn conversation** involves multiple back-and-forth interactions
between an agent and a human, which can allow the agent to gather additional
information from the human in a conversational manner.

This design pattern is useful in an LLM application consisting of multiple
agents. One or more agents may need to carry out multi-turn conversations with
a human, where the human provides input or feedback at different stages of the
conversation. For simplicity, the agent implementation below is illustrated as
a single node, but in reality it may be part of a larger graph consisting of
multiple nodes and include a conditional edge.

Using a human node per agentSharing human node across multiple agents

In this pattern, each agent has its own human node for collecting user input.
This can be achieved by either naming the human nodes with unique names (e.g.,
"human for agent 1", "human for agent 2") or by using subgraphs where a
subgraph contains a human node and an agent node.

    
    
    from langgraph.types import interrupt
    
    def human_input(state: State):
        human_message = interrupt("human_input")
        return {
            "messages": [
                {
                    "role": "human",
                    "content": human_message
                }
            ]
        }
    
    def agent(state: State):
        # Agent logic
        ...
    
    graph_builder.add_node("human_input", human_input)
    graph_builder.add_edge("human_input", "agent")
    graph = graph_builder.compile(checkpointer=checkpointer)
    
    # After running the graph and hitting the interrupt, the graph will pause.
    # Resume it with the human's input.
    graph.invoke(
        Command(resume="hello!"),
        config=thread_config
    )
    

API Reference: interrupt

In this pattern, a single human node is used to collect user input for
multiple agents. The active agent is determined from the state, so after human
input is collected, the graph can route to the correct agent.

    
    
    from langgraph.types import interrupt
    
    def human_node(state: MessagesState) -> Command[Literal["agent_1", "agent_2", ...]]:
        """A node for collecting user input."""
        user_input = interrupt(value="Ready for user input.")
    
        # Determine the **active agent** from the state, so 
        # we can route to the correct agent after collecting input.
        # For example, add a field to the state or use the last active agent.
        # or fill in `name` attribute of AI messages generated by the agents.
        active_agent = ... 
    
        return Command(
            update={
                "messages": [{
                    "role": "human",
                    "content": user_input,
                }]
            },
            goto=active_agent,
        )
    

API Reference: interrupt

See how to implement multi-turn conversations for a more detailed example.

### Validating human input¶

If you need to validate the input provided by the human within the graph
itself (rather than on the client side), you can achieve this by using
multiple interrupt calls within a single node.

    
    
    from langgraph.types import interrupt
    
    def human_node(state: State):
        """Human node with validation."""
        question = "What is your age?"
    
        while True:
            answer = interrupt(question)
    
            # Validate answer, if the answer isn't valid ask for input again.
            if not isinstance(answer, int) or answer < 0:
                question = f"'{answer} is not a valid age. What is your age?"
                answer = None
                continue
            else:
                # If the answer is valid, we can proceed.
                break
    
        print(f"The human in the loop is {answer} years old.")
        return {
            "age": answer
        }
    

API Reference: interrupt

## The `Command` primitive¶

When using the `interrupt` function, the graph will pause at the interrupt and
wait for user input.

Graph execution can be resumed using the Command primitive which can be passed
through the `invoke`, `ainvoke`, `stream` or `astream` methods.

The `Command` primitive provides several options to control and modify the
graph's state during resumption:

  1. **Pass a value to the`interrupt`**: Provide data, such as a user's response, to the graph using `Command(resume=value)`. Execution resumes from the beginning of the node where the `interrupt` was used, however, this time the `interrupt(...)` call will return the value passed in the `Command(resume=value)` instead of pausing the graph.
    
        # Resume graph execution with the user's input.
    graph.invoke(Command(resume={"age": "25"}), thread_config)
    

  2. **Update the graph state** : Modify the graph state using `Command(update=update)`. Note that resumption starts from the beginning of the node where the `interrupt` was used. Execution resumes from the beginning of the node where the `interrupt` was used, but with the updated state.
    
        # Update the graph state and resume.
    # You must provide a `resume` value if using an `interrupt`.
    graph.invoke(Command(update={"foo": "bar"}, resume="Let's go!!!"), thread_config)
    

By leveraging `Command`, you can resume graph execution, handle user inputs,
and dynamically adjust the graph's state.

## Using with `invoke` and `ainvoke`¶

When you use `stream` or `astream` to run the graph, you will receive an
`Interrupt` event that let you know the `interrupt` was triggered.

`invoke` and `ainvoke` do not return the interrupt information. To access this
information, you must use the get_state method to retrieve the graph state
after calling `invoke` or `ainvoke`.

    
    
    # Run the graph up to the interrupt 
    result = graph.invoke(inputs, thread_config)
    # Get the graph state to get interrupt information.
    state = graph.get_state(thread_config)
    # Print the state values
    print(state.values)
    # Print the pending tasks
    print(state.tasks)
    # Resume the graph with the user's input.
    graph.invoke(Command(resume={"age": "25"}), thread_config)
    
    
    
    {'foo': 'bar'} # State values
    (
        PregelTask(
            id='5d8ffc92-8011-0c9b-8b59-9d3545b7e553', 
            name='node_foo', 
            path=('__pregel_pull', 'node_foo'), 
            error=None, 
            interrupts=(Interrupt(value='value_in_interrupt', resumable=True, ns=['node_foo:5d8ffc92-8011-0c9b-8b59-9d3545b7e553'], when='during'),), state=None, 
            result=None
        ),
    ) # Pending tasks. interrupts 
    

## How does resuming from an interrupt work?¶

Warning

Resuming from an `interrupt` is **different** from Python's `input()`
function, where execution resumes from the exact point where the `input()`
function was called.

A critical aspect of using `interrupt` is understanding how resuming works.
When you resume execution after an `interrupt`, graph execution starts from
the **beginning** of the **graph node** where the last `interrupt` was
triggered.

**All** code from the beginning of the node to the `interrupt` will be re-
executed.

    
    
    counter = 0
    def node(state: State):
        # All the code from the beginning of the node to the interrupt will be re-executed
        # when the graph resumes.
        global counter
        counter += 1
        print(f"> Entered the node: {counter} # of times")
        # Pause the graph and wait for user input.
        answer = interrupt()
        print("The value of counter is:", counter)
        ...
    

Upon **resuming** the graph, the counter will be incremented a second time,
resulting in the following output:

    
    
    > Entered the node: 2 # of times
    The value of counter is: 2
    

## Common Pitfalls¶

### Side-effects¶

Place code with side effects, such as API calls, **after** the `interrupt` to
avoid duplication, as these are re-triggered every time the node is resumed.

Side effects before interrupt (BAD)Side effects after interrupt (OK)Side
effects in a separate node (OK)

This code will re-execute the API call another time when the node is resumed
from the `interrupt`.

This can be problematic if the API call is not idempotent or is just
expensive.

    
    
    from langgraph.types import interrupt
    
    def human_node(state: State):
        """Human node with validation."""
        api_call(...) # This code will be re-executed when the node is resumed.
        answer = interrupt(question)
    

API Reference: interrupt

    
    
    from langgraph.types import interrupt
    
    def human_node(state: State):
        """Human node with validation."""
    
        answer = interrupt(question)
    
        api_call(answer) # OK as it's after the interrupt
    

API Reference: interrupt

    
    
    from langgraph.types import interrupt
    
    def human_node(state: State):
        """Human node with validation."""
    
        answer = interrupt(question)
    
        return {
            "answer": answer
        }
    
    def api_call_node(state: State):
        api_call(...) # OK as it's in a separate node
    

API Reference: interrupt

### Subgraphs called as functions¶

When invoking a subgraph as a function, the **parent graph** will resume
execution from the **beginning of the node** where the subgraph was invoked
(and where an `interrupt` was triggered). Similarly, the **subgraph** , will
resume from the **beginning of the node** where the `interrupt()` function was
called.

For example,

    
    
    def node_in_parent_graph(state: State):
        some_code()  # <-- This will re-execute when the subgraph is resumed.
        # Invoke a subgraph as a function.
        # The subgraph contains an `interrupt` call.
        subgraph_result = subgraph.invoke(some_input)
        ...
    

**Example: Parent and Subgraph Execution Flow**

Say we have a parent graph with 3 nodes:

**Parent Graph** : `node_1` → `node_2` (subgraph call) → `node_3`

And the subgraph has 3 nodes, where the second node contains an `interrupt`:

**Subgraph** : `sub_node_1` → `sub_node_2` (`interrupt`) → `sub_node_3`

When resuming the graph, the execution will proceed as follows:

  1. **Skip`node_1`** in the parent graph (already executed, graph state was saved in snapshot).
  2. **Re-execute`node_2`** in the parent graph from the start.
  3. **Skip`sub_node_1`** in the subgraph (already executed, graph state was saved in snapshot).
  4. **Re-execute`sub_node_2`** in the subgraph from the beginning.
  5. Continue with `sub_node_3` and subsequent nodes.

Here is abbreviated example code that you can use to understand how subgraphs
work with interrupts. It counts the number of times each node is entered and
prints the count.

    
    
    import uuid
    from typing import TypedDict
    
    from langgraph.graph import StateGraph
    from langgraph.constants import START
    from langgraph.types import interrupt, Command
    from langgraph.checkpoint.memory import MemorySaver
    
    
    class State(TypedDict):
       """The graph state."""
       state_counter: int
    
    
    counter_node_in_subgraph = 0
    
    def node_in_subgraph(state: State):
       """A node in the sub-graph."""
       global counter_node_in_subgraph
       counter_node_in_subgraph += 1  # This code will **NOT** run again!
       print(f"Entered `node_in_subgraph` a total of {counter_node_in_subgraph} times")
    
    counter_human_node = 0
    
    def human_node(state: State):
       global counter_human_node
       counter_human_node += 1 # This code will run again!
       print(f"Entered human_node in sub-graph a total of {counter_human_node} times")
       answer = interrupt("what is your name?")
       print(f"Got an answer of {answer}")
    
    
    checkpointer = MemorySaver()
    
    subgraph_builder = StateGraph(State)
    subgraph_builder.add_node("some_node", node_in_subgraph)
    subgraph_builder.add_node("human_node", human_node)
    subgraph_builder.add_edge(START, "some_node")
    subgraph_builder.add_edge("some_node", "human_node")
    subgraph = subgraph_builder.compile(checkpointer=checkpointer)
    
    
    counter_parent_node = 0
    
    def parent_node(state: State):
       """This parent node will invoke the subgraph."""
       global counter_parent_node
    
       counter_parent_node += 1 # This code will run again on resuming!
       print(f"Entered `parent_node` a total of {counter_parent_node} times")
    
       # Please note that we're intentionally incrementing the state counter
       # in the graph state as well to demonstrate that the subgraph update
       # of the same key will not conflict with the parent graph (until
       subgraph_state = subgraph.invoke(state)
       return subgraph_state
    
    
    builder = StateGraph(State)
    builder.add_node("parent_node", parent_node)
    builder.add_edge(START, "parent_node")
    
    # A checkpointer must be enabled for interrupts to work!
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    
    config = {
       "configurable": {
          "thread_id": uuid.uuid4(),
       }
    }
    
    for chunk in graph.stream({"state_counter": 1}, config):
       print(chunk)
    
    print('--- Resuming ---')
    
    for chunk in graph.stream(Command(resume="35"), config):
       print(chunk)
    

API Reference: StateGraph | START | interrupt | Command | MemorySaver

This will print out

    
    
    --- First invocation ---
    In parent node: {'foo': 'bar'}
    Entered `parent_node` a total of 1 times
    Entered `node_in_subgraph` a total of 1 times
    Entered human_node in sub-graph a total of 1 times
    {'__interrupt__': (Interrupt(value='what is your name?', resumable=True, ns=['parent_node:0b23d72f-aaba-0329-1a59-ca4f3c8bad3b', 'human_node:25df717c-cb80-57b0-7410-44e20aac8f3c'], when='during'),)}
    
    --- Resuming ---
    In parent node: {'foo': 'bar'}
    Entered `parent_node` a total of 2 times
    Entered human_node in sub-graph a total of 2 times
    Got an answer of 35
    {'parent_node': None} 
    

### Using multiple interrupts¶

Using multiple interrupts within a **single** node can be helpful for patterns
like validating human input. However, using multiple interrupts in the same
node can lead to unexpected behavior if not handled carefully.

When a node contains multiple interrupt calls, LangGraph keeps a list of
resume values specific to the task executing the node. Whenever execution
resumes, it starts at the beginning of the node. For each interrupt
encountered, LangGraph checks if a matching value exists in the task's resume
list. Matching is **strictly index-based** , so the order of interrupt calls
within the node is critical.

To avoid issues, refrain from dynamically changing the node's structure
between executions. This includes adding, removing, or reordering interrupt
calls, as such changes can result in mismatched indices. These problems often
arise from unconventional patterns, such as mutating state via
`Command(resume=..., update=SOME_STATE_MUTATION)` or relying on global
variables to modify the node’s structure dynamically.

Example of incorrect code

    
    
    import uuid
    from typing import TypedDict, Optional
    
    from langgraph.graph import StateGraph
    from langgraph.constants import START 
    from langgraph.types import interrupt, Command
    from langgraph.checkpoint.memory import MemorySaver
    
    
    class State(TypedDict):
        """The graph state."""
    
        age: Optional[str]
        name: Optional[str]
    
    
    def human_node(state: State):
        if not state.get('name'):
            name = interrupt("what is your name?")
        else:
            name = "N/A"
    
        if not state.get('age'):
            age = interrupt("what is your age?")
        else:
            age = "N/A"
    
        print(f"Name: {name}. Age: {age}")
    
        return {
            "age": age,
            "name": name,
        }
    
    
    builder = StateGraph(State)
    builder.add_node("human_node", human_node)
    builder.add_edge(START, "human_node")
    
    # A checkpointer must be enabled for interrupts to work!
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    
    config = {
        "configurable": {
            "thread_id": uuid.uuid4(),
        }
    }
    
    for chunk in graph.stream({"age": None, "name": None}, config):
        print(chunk)
    
    for chunk in graph.stream(Command(resume="John", update={"name": "foo"}), config):
        print(chunk)
    

API Reference: StateGraph | START | interrupt | Command | MemorySaver
    
    
    {'__interrupt__': (Interrupt(value='what is your name?', resumable=True, ns=['human_node:3a007ef9-c30d-c357-1ec1-86a1a70d8fba'], when='during'),)}
    Name: N/A. Age: John
    {'human_node': {'age': 'John', 'name': 'N/A'}}
    

## Additional Resources 📚¶

  * **Conceptual Guide: Persistence** : Read the persistence guide for more context on replaying.
  * **How to Guides: Human-in-the-loop** : Learn how to implement human-in-the-loop workflows in LangGraph.
  * **How to implement multi-turn conversations** : Learn how to implement multi-turn conversations in LangGraph.

## Comments

Back to top

Previous

Multi-agent Systems

Next

Persistence

Made with  Material for MkDocs Insiders



Skip to content

To learn more about LangGraph, check out our first LangChain Academy course,
_Introduction to LangGraph_ , available for free here.

![logo](./LangGraph Glossary_files/wordmark_dark.svg) ![logo](./LangGraph
Glossary_files/wordmark_light.svg)

LangGraph Glossary

Type to start searching

GitHub

  * 0.2.62
  * 7.9k
  * 1.3k

  * Home 
  * Tutorials 
  * How-to Guides 
  * Conceptual Guides 
  * Reference 

![logo](./LangGraph Glossary_files/wordmark_dark.svg) ![logo](./LangGraph
Glossary_files/wordmark_light.svg)

GitHub

  * 0.2.62
  * 7.9k
  * 1.3k

  * Home 
  * Tutorials 
  * How-to Guides 
  * Conceptual Guides 

Conceptual Guides

    * LangGraph  LangGraph 
      * LangGraph 
      * Why LangGraph? 
      * LangGraph Glossary  LangGraph Glossary  Table of contents 
        * Graphs 
          * StateGraph 
          * MessageGraph 
          * Compiling your graph 
        * State 
          * Schema 
            * Multiple schemas 
          * Reducers 
            * Default Reducer 
          * Working with Messages in Graph State 
            * Why use messages? 
            * Using Messages in your Graph 
            * Serialization 
            * MessagesState 
        * Nodes 
          * START Node 
          * END Node 
        * Edges 
          * Normal Edges 
          * Conditional Edges 
          * Entry Point 
          * Conditional Entry Point 
        * Send 
        * Command 
          * When should I use Command instead of conditional edges? 
          * Using inside tools 
          * Human-in-the-loop 
        * Persistence 
        * Threads 
        * Storage 
        * Graph Migrations 
        * Configuration 
          * Recursion Limit 
        * interrupt 
        * Breakpoints 
        * Subgraphs 
          * As a compiled graph 
          * As a function 
        * Visualization 
        * Streaming 
      * Agent architectures 
      * Multi-agent Systems 
      * Human-in-the-loop 
      * Persistence 
      * Memory 
      * Streaming 
      * FAQ 
    * LangGraph Platform  LangGraph Platform 
      * LangGraph Platform 
      * High Level 
      * Components 
      * LangGraph Server 
      * Deployment Options 
  * Reference 

Table of contents

  * Graphs 
    * StateGraph 
    * MessageGraph 
    * Compiling your graph 
  * State 
    * Schema 
      * Multiple schemas 
    * Reducers 
      * Default Reducer 
    * Working with Messages in Graph State 
      * Why use messages? 
      * Using Messages in your Graph 
      * Serialization 
      * MessagesState 
  * Nodes 
    * START Node 
    * END Node 
  * Edges 
    * Normal Edges 
    * Conditional Edges 
    * Entry Point 
    * Conditional Entry Point 
  * Send 
  * Command 
    * When should I use Command instead of conditional edges? 
    * Using inside tools 
    * Human-in-the-loop 
  * Persistence 
  * Threads 
  * Storage 
  * Graph Migrations 
  * Configuration 
    * Recursion Limit 
  * interrupt 
  * Breakpoints 
  * Subgraphs 
    * As a compiled graph 
    * As a function 
  * Visualization 
  * Streaming 

  1. Home 
  2. Conceptual Guides 
  3. LangGraph 

# LangGraph Glossary¶

## Graphs¶

At its core, LangGraph models agent workflows as graphs. You define the
behavior of your agents using three key components:

  1. `State`: A shared data structure that represents the current snapshot of your application. It can be any Python type, but is typically a `TypedDict` or Pydantic `BaseModel`.

  2. `Nodes`: Python functions that encode the logic of your agents. They receive the current `State` as input, perform some computation or side-effect, and return an updated `State`.

  3. `Edges`: Python functions that determine which `Node` to execute next based on the current `State`. They can be conditional branches or fixed transitions.

By composing `Nodes` and `Edges`, you can create complex, looping workflows
that evolve the `State` over time. The real power, though, comes from how
LangGraph manages that `State`. To emphasize: `Nodes` and `Edges` are nothing
more than Python functions - they can contain an LLM or just good ol' Python
code.

In short: _nodes do the work. edges tell what to do next_.

LangGraph's underlying graph algorithm uses message passing to define a
general program. When a Node completes its operation, it sends messages along
one or more edges to other node(s). These recipient nodes then execute their
functions, pass the resulting messages to the next set of nodes, and the
process continues. Inspired by Google's Pregel system, the program proceeds in
discrete "super-steps."

A super-step can be considered a single iteration over the graph nodes. Nodes
that run in parallel are part of the same super-step, while nodes that run
sequentially belong to separate super-steps. At the start of graph execution,
all nodes begin in an `inactive` state. A node becomes `active` when it
receives a new message (state) on any of its incoming edges (or "channels").
The active node then runs its function and responds with updates. At the end
of each super-step, nodes with no incoming messages vote to `halt` by marking
themselves as `inactive`. The graph execution terminates when all nodes are
`inactive` and no messages are in transit.

### StateGraph¶

The `StateGraph` class is the main graph class to use. This is parameterized
by a user defined `State` object.

### MessageGraph¶

The `MessageGraph` class is a special type of graph. The `State` of a
`MessageGraph` is ONLY a list of messages. This class is rarely used except
for chatbots, as most applications require the `State` to be more complex than
a list of messages.

### Compiling your graph¶

To build your graph, you first define the state, you then add nodes and edges,
and then you compile it. What exactly is compiling your graph and why is it
needed?

Compiling is a pretty simple step. It provides a few basic checks on the
structure of your graph (no orphaned nodes, etc). It is also where you can
specify runtime args like checkpointers and breakpoints. You compile your
graph by just calling the `.compile` method:

    
    
    graph = graph_builder.compile(...)
    

You **MUST** compile your graph before you can use it.

## State¶

The first thing you do when you define a graph is define the `State` of the
graph. The `State` consists of the schema of the graph as well as `reducer`
functions which specify how to apply updates to the state. The schema of the
`State` will be the input schema to all `Nodes` and `Edges` in the graph, and
can be either a `TypedDict` or a `Pydantic` model. All `Nodes` will emit
updates to the `State` which are then applied using the specified `reducer`
function.

### Schema¶

The main documented way to specify the schema of a graph is by using
`TypedDict`. However, we also support using a Pydantic BaseModel as your graph
state to add **default values** and additional data validation.

By default, the graph will have the same input and output schemas. If you want
to change this, you can also specify explicit input and output schemas
directly. This is useful when you have a lot of keys, and some are explicitly
for input and others for output. See the notebook here for how to use.

#### Multiple schemas¶

Typically, all graph nodes communicate with a single schema. This means that
they will read and write to the same state channels. But, there are cases
where we want more control over this:

  * Internal nodes can pass information that is not required in the graph's input / output.
  * We may also want to use different input / output schemas for the graph. The output might, for example, only contain a single relevant output key.

It is possible to have nodes write to private state channels inside the graph
for internal node communication. We can simply define a private schema,
`PrivateState`. See this notebook for more detail.

It is also possible to define explicit input and output schemas for a graph.
In these cases, we define an "internal" schema that contains _all_ keys
relevant to graph operations. But, we also define `input` and `output` schemas
that are sub-sets of the "internal" schema to constrain the input and output
of the graph. See this notebook for more detail.

Let's look at an example:

    
    
    class InputState(TypedDict):
        user_input: str
    
    class OutputState(TypedDict):
        graph_output: str
    
    class OverallState(TypedDict):
        foo: str
        user_input: str
        graph_output: str
    
    class PrivateState(TypedDict):
        bar: str
    
    def node_1(state: InputState) -> OverallState:
        # Write to OverallState
        return {"foo": state["user_input"] + " name"}
    
    def node_2(state: OverallState) -> PrivateState:
        # Read from OverallState, write to PrivateState
        return {"bar": state["foo"] + " is"}
    
    def node_3(state: PrivateState) -> OutputState:
        # Read from PrivateState, write to OutputState
        return {"graph_output": state["bar"] + " Lance"}
    
    builder = StateGraph(OverallState,input=InputState,output=OutputState)
    builder.add_node("node_1", node_1)
    builder.add_node("node_2", node_2)
    builder.add_node("node_3", node_3)
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "node_2")
    builder.add_edge("node_2", "node_3")
    builder.add_edge("node_3", END)
    
    graph = builder.compile()
    graph.invoke({"user_input":"My"})
    {'graph_output': 'My name is Lance'}
    

There are two subtle and important points to note here:

  1. We pass `state: InputState` as the input schema to `node_1`. But, we write out to `foo`, a channel in `OverallState`. How can we write out to a state channel that is not included in the input schema? This is because a node _can write to any state channel in the graph state._ The graph state is the union of of the state channels defined at initialization, which includes `OverallState` and the filters `InputState` and `OutputState`.

  2. We initialize the graph with `StateGraph(OverallState,input=InputState,output=OutputState)`. So, how can we write to `PrivateState` in `node_2`? How does the graph gain access to this schema if it was not passed in the `StateGraph` initialization? We can do this because _nodes can also declare additional state channels_ as long as the state schema definition exists. In this case, the `PrivateState` schema is defined, so we can add `bar` as a new state channel in the graph and write to it.

### Reducers¶

Reducers are key to understanding how updates from nodes are applied to the
`State`. Each key in the `State` has its own independent reducer function. If
no reducer function is explicitly specified then it is assumed that all
updates to that key should override it. There are a few different types of
reducers, starting with the default type of reducer:

#### Default Reducer¶

These two examples show how to use the default reducer:

**Example A:**

    
    
    from typing_extensions import TypedDict
    
    class State(TypedDict):
        foo: int
        bar: list[str]
    

In this example, no reducer functions are specified for any key. Let's assume
the input to the graph is `{"foo": 1, "bar": ["hi"]}`. Let's then assume the
first `Node` returns `{"foo": 2}`. This is treated as an update to the state.
Notice that the `Node` does not need to return the whole `State` schema - just
an update. After applying this update, the `State` would then be `{"foo": 2,
"bar": ["hi"]}`. If the second node returns `{"bar": ["bye"]}` then the
`State` would then be `{"foo": 2, "bar": ["bye"]}`

**Example B:**

    
    
    from typing import Annotated
    from typing_extensions import TypedDict
    from operator import add
    
    class State(TypedDict):
        foo: int
        bar: Annotated[list[str], add]
    

In this example, we've used the `Annotated` type to specify a reducer function
(`operator.add`) for the second key (`bar`). Note that the first key remains
unchanged. Let's assume the input to the graph is `{"foo": 1, "bar": ["hi"]}`.
Let's then assume the first `Node` returns `{"foo": 2}`. This is treated as an
update to the state. Notice that the `Node` does not need to return the whole
`State` schema - just an update. After applying this update, the `State` would
then be `{"foo": 2, "bar": ["hi"]}`. If the second node returns `{"bar":
["bye"]}` then the `State` would then be `{"foo": 2, "bar": ["hi", "bye"]}`.
Notice here that the `bar` key is updated by adding the two lists together.

### Working with Messages in Graph State¶

#### Why use messages?¶

Most modern LLM providers have a chat model interface that accepts a list of
messages as input. LangChain's `ChatModel` in particular accepts a list of
`Message` objects as inputs. These messages come in a variety of forms such as
`HumanMessage` (user input) or `AIMessage` (LLM response). To read more about
what message objects are, please refer to this conceptual guide.

#### Using Messages in your Graph¶

In many cases, it is helpful to store prior conversation history as a list of
messages in your graph state. To do so, we can add a key (channel) to the
graph state that stores a list of `Message` objects and annotate it with a
reducer function (see `messages` key in the example below). The reducer
function is vital to telling the graph how to update the list of `Message`
objects in the state with each state update (for example, when a node sends an
update). If you don't specify a reducer, every state update will overwrite the
list of messages with the most recently provided value. If you wanted to
simply append messages to the existing list, you could use `operator.add` as a
reducer.

However, you might also want to manually update messages in your graph state
(e.g. human-in-the-loop). If you were to use `operator.add`, the manual state
updates you send to the graph would be appended to the existing list of
messages, instead of updating existing messages. To avoid that, you need a
reducer that can keep track of message IDs and overwrite existing messages, if
updated. To achieve this, you can use the prebuilt `add_messages` function.
For brand new messages, it will simply append to existing list, but it will
also handle the updates for existing messages correctly.

#### Serialization¶

In addition to keeping track of message IDs, the `add_messages` function will
also try to deserialize messages into LangChain `Message` objects whenever a
state update is received on the `messages` channel. See more information on
LangChain serialization/deserialization here. This allows sending graph inputs
/ state updates in the following format:

    
    
    # this is supported
    {"messages": [HumanMessage(content="message")]}
    
    # and this is also supported
    {"messages": [{"type": "human", "content": "message"}]}
    

Since the state updates are always deserialized into LangChain `Messages` when
using `add_messages`, you should use dot notation to access message
attributes, like `state["messages"][-1].content`. Below is an example of a
graph that uses `add_messages` as it's reducer function.

    
    
    from langchain_core.messages import AnyMessage
    from langgraph.graph.message import add_messages
    from typing import Annotated
    from typing_extensions import TypedDict
    
    class GraphState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
    

API Reference: add_messages

#### MessagesState¶

Since having a list of messages in your state is so common, there exists a
prebuilt state called `MessagesState` which makes it easy to use messages.
`MessagesState` is defined with a single `messages` key which is a list of
`AnyMessage` objects and uses the `add_messages` reducer. Typically, there is
more state to track than just messages, so we see people subclass this state
and add more fields, like:

    
    
    from langgraph.graph import MessagesState
    
    class State(MessagesState):
        documents: list[str]
    

## Nodes¶

In LangGraph, nodes are typically python functions (sync or async) where the
**first** positional argument is the state, and (optionally), the **second**
positional argument is a "config", containing optional configurable parameters
(such as a `thread_id`).

Similar to `NetworkX`, you add these nodes to a graph using the add_node
method:

    
    
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph import StateGraph
    
    builder = StateGraph(dict)
    
    
    def my_node(state: dict, config: RunnableConfig):
        print("In node: ", config["configurable"]["user_id"])
        return {"results": f"Hello, {state['input']}!"}
    
    
    # The second argument is optional
    def my_other_node(state: dict):
        return state
    
    
    builder.add_node("my_node", my_node)
    builder.add_node("other_node", my_other_node)
    ...
    

API Reference: RunnableConfig | StateGraph

Behind the scenes, functions are converted to RunnableLambda's, which add
batch and async support to your function, along with native tracing and
debugging.

If you add a node to graph without specifying a name, it will be given a
default name equivalent to the function name.

    
    
    builder.add_node(my_node)
    # You can then create edges to/from this node by referencing it as `"my_node"`
    

### `START` Node¶

The `START` Node is a special node that represents the node sends user input
to the graph. The main purpose for referencing this node is to determine which
nodes should be called first.

    
    
    from langgraph.graph import START
    
    graph.add_edge(START, "node_a")
    

API Reference: START

### `END` Node¶

The `END` Node is a special node that represents a terminal node. This node is
referenced when you want to denote which edges have no actions after they are
done.

    
    
    from langgraph.graph import END
    
    graph.add_edge("node_a", END)
    

## Edges¶

Edges define how the logic is routed and how the graph decides to stop. This
is a big part of how your agents work and how different nodes communicate with
each other. There are a few key types of edges:

  * Normal Edges: Go directly from one node to the next.
  * Conditional Edges: Call a function to determine which node(s) to go to next.
  * Entry Point: Which node to call first when user input arrives.
  * Conditional Entry Point: Call a function to determine which node(s) to call first when user input arrives.

A node can have MULTIPLE outgoing edges. If a node has multiple out-going
edges, **all** of those destination nodes will be executed in parallel as a
part of the next superstep.

### Normal Edges¶

If you **always** want to go from node A to node B, you can use the add_edge
method directly.

    
    
    graph.add_edge("node_a", "node_b")
    

### Conditional Edges¶

If you want to **optionally** route to 1 or more edges (or optionally
terminate), you can use the add_conditional_edges method. This method accepts
the name of a node and a "routing function" to call after that node is
executed:

    
    
    graph.add_conditional_edges("node_a", routing_function)
    

Similar to nodes, the `routing_function` accept the current `state` of the
graph and return a value.

By default, the return value `routing_function` is used as the name of the
node (or a list of nodes) to send the state to next. All those nodes will be
run in parallel as a part of the next superstep.

You can optionally provide a dictionary that maps the `routing_function`'s
output to the name of the next node.

    
    
    graph.add_conditional_edges("node_a", routing_function, {True: "node_b", False: "node_c"})
    

Tip

Use `Command` instead of conditional edges if you want to combine state
updates and routing in a single function.

### Entry Point¶

The entry point is the first node(s) that are run when the graph starts. You
can use the `add_edge` method from the virtual `START` node to the first node
to execute to specify where to enter the graph.

    
    
    from langgraph.graph import START
    
    graph.add_edge(START, "node_a")
    

API Reference: START

### Conditional Entry Point¶

A conditional entry point lets you start at different nodes depending on
custom logic. You can use `add_conditional_edges` from the virtual `START`
node to accomplish this.

    
    
    from langgraph.graph import START
    
    graph.add_conditional_edges(START, routing_function)
    

API Reference: START

You can optionally provide a dictionary that maps the `routing_function`'s
output to the name of the next node.

    
    
    graph.add_conditional_edges(START, routing_function, {True: "node_b", False: "node_c"})
    

## `Send`¶

By default, `Nodes` and `Edges` are defined ahead of time and operate on the
same shared state. However, there can be cases where the exact edges are not
known ahead of time and/or you may want different versions of `State` to exist
at the same time. A common of example of this is with `map-reduce` design
patterns. In this design pattern, a first node may generate a list of objects,
and you may want to apply some other node to all those objects. The number of
objects may be unknown ahead of time (meaning the number of edges may not be
known) and the input `State` to the downstream `Node` should be different (one
for each generated object).

To support this design pattern, LangGraph supports returning `Send` objects
from conditional edges. `Send` takes two arguments: first is the name of the
node, and second is the state to pass to that node.

    
    
    def continue_to_jokes(state: OverallState):
        return [Send("generate_joke", {"subject": s}) for s in state['subjects']]
    
    graph.add_conditional_edges("node_a", continue_to_jokes)
    

## `Command`¶

It can be useful to combine control flow (edges) and state updates (nodes).
For example, you might want to BOTH perform state updates AND decide which
node to go to next in the SAME node. LangGraph provides a way to do so by
returning a `Command` object from node functions:

    
    
    def my_node(state: State) -> Command[Literal["my_other_node"]]:
        return Command(
            # state update
            update={"foo": "bar"},
            # control flow
            goto="my_other_node"
        )
    

With `Command` you can also achieve dynamic control flow behavior (identical
to conditional edges):

    
    
    def my_node(state: State) -> Command[Literal["my_other_node"]]:
        if state["foo"] == "bar":
            return Command(update={"foo": "baz"}, goto="my_other_node")
    

Important

When returning `Command` in your node functions, you must add return type
annotations with the list of node names the node is routing to, e.g.
`Command[Literal["my_other_node"]]`. This is necessary for the graph rendering
and tells LangGraph that `my_node` can navigate to `my_other_node`.

Check out this how-to guide for an end-to-end example of how to use `Command`.

### When should I use Command instead of conditional edges?¶

Use `Command` when you need to **both** update the graph state **and** route
to a different node. For example, when implementing multi-agent handoffs where
it's important to route to a different agent and pass some information to that
agent.

Use conditional edges to route between nodes conditionally without updating
the state.

### Using inside tools¶

A common use case is updating graph state from inside a tool. For example, in
a customer support application you might want to look up customer information
based on their account number or ID in the beginning of the conversation. To
update the graph state from the tool, you can return
`Command(update={"my_custom_key": "foo", "messages": [...]})` from the tool:

    
    
    @tool
    def lookup_user_info(tool_call_id: Annotated[str, InjectedToolCallId], config: RunnableConfig):
        """Use this to look up user information to better assist them with their questions."""
        user_info = get_user_info(config.get("configurable", {}).get("user_id"))
        return Command(
            update={
                # update the state keys
                "user_info": user_info,
                # update the message history
                "messages": [ToolMessage("Successfully looked up user information", tool_call_id=tool_call_id)]
            }
        )
    

Important

You MUST include `messages` (or any state key used for the message history) in
`Command.update` when returning `Command` from a tool and the list of messages
in `messages` MUST contain a `ToolMessage`. This is necessary for the
resulting message history to be valid (LLM providers require AI messages with
tool calls to be followed by the tool result messages).

If you are using tools that update state via `Command`, we recommend using
prebuilt `ToolNode` which automatically handles tools returning `Command`
objects and propagates them to the graph state. If you're writing a custom
node that calls tools, you would need to manually propagate `Command` objects
returned by the tools as the update from node.

### Human-in-the-loop¶

`Command` is an important part of human-in-the-loop workflows: when using
`interrupt()` to collect user input, `Command` is then used to supply the
input and resume execution via `Command(resume="User input")`. Check out this
conceptual guide for more information.

## Persistence¶

LangGraph provides built-in persistence for your agent's state using
checkpointers. Checkpointers save snapshots of the graph state at every
superstep, allowing resumption at any time. This enables features like human-
in-the-loop interactions, memory management, and fault-tolerance. You can even
directly manipulate a graph's state after its execution using the appropriate
`get` and `update` methods. For more details, see the persistence conceptual
guide.

## Threads¶

Threads in LangGraph represent individual sessions or conversations between
your graph and a user. When using checkpointing, turns in a single
conversation (and even steps within a single graph execution) are organized by
a unique thread ID.

## Storage¶

LangGraph provides built-in document storage through the BaseStore interface.
Unlike checkpointers, which save state by thread ID, stores use custom
namespaces for organizing data. This enables cross-thread persistence,
allowing agents to maintain long-term memories, learn from past interactions,
and accumulate knowledge over time. Common use cases include storing user
profiles, building knowledge bases, and managing global preferences across all
threads.

## Graph Migrations¶

LangGraph can easily handle migrations of graph definitions (nodes, edges, and
state) even when using a checkpointer to track state.

  * For threads at the end of the graph (i.e. not interrupted) you can change the entire topology of the graph (i.e. all nodes and edges, remove, add, rename, etc)
  * For threads currently interrupted, we support all topology changes other than renaming / removing nodes (as that thread could now be about to enter a node that no longer exists) -- if this is a blocker please reach out and we can prioritize a solution.
  * For modifying state, we have full backwards and forwards compatibility for adding and removing keys
  * State keys that are renamed lose their saved state in existing threads
  * State keys whose types change in incompatible ways could currently cause issues in threads with state from before the change -- if this is a blocker please reach out and we can prioritize a solution.

## Configuration¶

When creating a graph, you can also mark that certain parts of the graph are
configurable. This is commonly done to enable easily switching between models
or system prompts. This allows you to create a single "cognitive architecture"
(the graph) but have multiple different instance of it.

You can optionally specify a `config_schema` when creating a graph.

    
    
    class ConfigSchema(TypedDict):
        llm: str
    
    graph = StateGraph(State, config_schema=ConfigSchema)
    

You can then pass this configuration into the graph using the `configurable`
config field.

    
    
    config = {"configurable": {"llm": "anthropic"}}
    
    graph.invoke(inputs, config=config)
    

You can then access and use this configuration inside a node:

    
    
    def node_a(state, config):
        llm_type = config.get("configurable", {}).get("llm", "openai")
        llm = get_llm(llm_type)
        ...
    

See this guide for a full breakdown on configuration.

### Recursion Limit¶

The recursion limit sets the maximum number of super-steps the graph can
execute during a single execution. Once the limit is reached, LangGraph will
raise `GraphRecursionError`. By default this value is set to 25 steps. The
recursion limit can be set on any graph at runtime, and is passed to
`.invoke`/`.stream` via the config dictionary. Importantly, `recursion_limit`
is a standalone `config` key and should not be passed inside the
`configurable` key as all other user-defined configuration. See the example
below:

    
    
    graph.invoke(inputs, config={"recursion_limit": 5, "configurable":{"llm": "anthropic"}})
    

Read this how-to to learn more about how the recursion limit works.

## `interrupt`¶

Use the interrupt function to **pause** the graph at specific points to
collect user input. The `interrupt` function surfaces interrupt information to
the client, allowing the developer to collect user input, validate the graph
state, or make decisions before resuming execution.

    
    
    from langgraph.types import interrupt
    
    def human_approval_node(state: State):
        ...
        answer = interrupt(
            # This value will be sent to the client.
            # It can be any JSON serializable value.
            {"question": "is it ok to continue?"},
        )
        ...
    

API Reference: interrupt

Resuming the graph is done by passing a `Command` object to the graph with the
`resume` key set to the value returned by the `interrupt` function.

Read more about how the `interrupt` is used for **human-in-the-loop**
workflows in the Human-in-the-loop conceptual guide.

## Breakpoints¶

Breakpoints pause graph execution at specific points and enable stepping
through execution step by step. Breakpoints are powered by LangGraph's
**persistence layer** , which saves the state after each graph step.
Breakpoints can also be used to enable **human-in-the-loop** workflows, though
we recommend using the `interrupt` function for this purpose.

Read more about breakpoints in the Breakpoints conceptual guide.

## Subgraphs¶

A subgraph is a graph that is used as a node in another graph. This is nothing
more than the age-old concept of encapsulation, applied to LangGraph. Some
reasons for using subgraphs are:

  * building multi-agent systems

  * when you want to reuse a set of nodes in multiple graphs, which maybe share some state, you can define them once in a subgraph and then use them in multiple parent graphs

  * when you want different teams to work on different parts of the graph independently, you can define each part as a subgraph, and as long as the subgraph interface (the input and output schemas) is respected, the parent graph can be built without knowing any details of the subgraph

There are two ways to add subgraphs to a parent graph:

  * add a node with the compiled subgraph: this is useful when the parent graph and the subgraph share state keys and you don't need to transform state on the way in or out

    
    
    builder.add_node("subgraph", subgraph_builder.compile())
    

  * add a node with a function that invokes the subgraph: this is useful when the parent graph and the subgraph have different state schemas and you need to transform state before or after calling the subgraph

    
    
    subgraph = subgraph_builder.compile()
    
    def call_subgraph(state: State):
        return subgraph.invoke({"subgraph_key": state["parent_key"]})
    
    builder.add_node("subgraph", call_subgraph)
    

Let's take a look at examples for each.

### As a compiled graph¶

The simplest way to create subgraph nodes is by using a compiled subgraph
directly. When doing so, it is **important** that the parent graph and the
subgraph state schemas share at least one key which they can use to
communicate. If your graph and subgraph do not share any keys, you should use
write a function invoking the subgraph instead.

Note

If you pass extra keys to the subgraph node (i.e., in addition to the shared
keys), they will be ignored by the subgraph node. Similarly, if you return
extra keys from the subgraph, they will be ignored by the parent graph.

    
    
    from langgraph.graph import StateGraph
    from typing import TypedDict
    
    class State(TypedDict):
        foo: str
    
    class SubgraphState(TypedDict):
        foo: str  # note that this key is shared with the parent graph state
        bar: str
    
    # Define subgraph
    def subgraph_node(state: SubgraphState):
        # note that this subgraph node can communicate with the parent graph via the shared "foo" key
        return {"foo": state["foo"] + "bar"}
    
    subgraph_builder = StateGraph(SubgraphState)
    subgraph_builder.add_node(subgraph_node)
    ...
    subgraph = subgraph_builder.compile()
    
    # Define parent graph
    builder = StateGraph(State)
    builder.add_node("subgraph", subgraph)
    ...
    graph = builder.compile()
    

API Reference: StateGraph

### As a function¶

You might want to define a subgraph with a completely different schema. In
this case, you can create a node function that invokes the subgraph. This
function will need to transform the input (parent) state to the subgraph state
before invoking the subgraph, and transform the results back to the parent
state before returning the state update from the node.

    
    
    class State(TypedDict):
        foo: str
    
    class SubgraphState(TypedDict):
        # note that none of these keys are shared with the parent graph state
        bar: str
        baz: str
    
    # Define subgraph
    def subgraph_node(state: SubgraphState):
        return {"bar": state["bar"] + "baz"}
    
    subgraph_builder = StateGraph(SubgraphState)
    subgraph_builder.add_node(subgraph_node)
    ...
    subgraph = subgraph_builder.compile()
    
    # Define parent graph
    def node(state: State):
        # transform the state to the subgraph state
        response = subgraph.invoke({"bar": state["foo"]})
        # transform response back to the parent state
        return {"foo": response["bar"]}
    
    builder = StateGraph(State)
    # note that we are using `node` function instead of a compiled subgraph
    builder.add_node(node)
    ...
    graph = builder.compile()
    

## Visualization¶

It's often nice to be able to visualize graphs, especially as they get more
complex. LangGraph comes with several built-in ways to visualize graphs. See
this how-to guide for more info.

## Streaming¶

LangGraph is built with first class support for streaming, including streaming
updates from graph nodes during the execution, streaming tokens from LLM calls
and more. See this conceptual guide for more information.

## Comments

Back to top

Previous

Why LangGraph?

Next

Agent architectures

Made with  Material for MkDocs Insiders



Skip to content

To learn more about LangGraph, check out our first LangChain Academy course,
_Introduction to LangGraph_ , available for free here.

![logo](./Memory_files/wordmark_dark.svg)
![logo](./Memory_files/wordmark_light.svg)

Memory

Type to start searching

GitHub

  * 0.2.62
  * 7.9k
  * 1.3k

  * Home 
  * Tutorials 
  * How-to Guides 
  * Conceptual Guides 
  * Reference 

![logo](./Memory_files/wordmark_dark.svg)
![logo](./Memory_files/wordmark_light.svg)

GitHub

  * 0.2.62
  * 7.9k
  * 1.3k

  * Home 
  * Tutorials 
  * How-to Guides 
  * Conceptual Guides 

Conceptual Guides

    * LangGraph  LangGraph 
      * LangGraph 
      * Why LangGraph? 
      * LangGraph Glossary 
      * Agent architectures 
      * Multi-agent Systems 
      * Human-in-the-loop 
      * Persistence 
      * Memory  Memory  Table of contents 
        * What is Memory? 
        * Short-term memory 
          * Managing long conversation history 
          * Editing message lists 
          * Summarizing past conversations 
          * Knowing when to remove messages 
        * Long-term memory 
          * Storing memories 
          * Framework for thinking about long-term memory 
        * Memory types 
          * Semantic Memory 
            * Profile 
            * Collection 
          * Episodic Memory 
          * Procedural Memory 
        * Writing memories 
          * Writing memories in the hot path 
          * Writing memories in the background 
      * Streaming 
      * FAQ 
    * LangGraph Platform  LangGraph Platform 
      * LangGraph Platform 
      * High Level 
      * Components 
      * LangGraph Server 
      * Deployment Options 
  * Reference 

Table of contents

  * What is Memory? 
  * Short-term memory 
    * Managing long conversation history 
    * Editing message lists 
    * Summarizing past conversations 
    * Knowing when to remove messages 
  * Long-term memory 
    * Storing memories 
    * Framework for thinking about long-term memory 
  * Memory types 
    * Semantic Memory 
      * Profile 
      * Collection 
    * Episodic Memory 
    * Procedural Memory 
  * Writing memories 
    * Writing memories in the hot path 
    * Writing memories in the background 

  1. Home 
  2. Conceptual Guides 
  3. LangGraph 

# Memory¶

## What is Memory?¶

Memory is a cognitive function that allows people to store, retrieve, and use
information to understand their present and future. Consider the frustration
of working with a colleague who forgets everything you tell them, requiring
constant repetition! As AI agents undertake more complex tasks involving
numerous user interactions, equipping them with memory becomes equally crucial
for efficiency and user satisfaction. With memory, agents can learn from
feedback and adapt to users' preferences. This guide covers two types of
memory based on recall scope:

**Short-term memory** , or thread-scoped memory, can be recalled at any time
**from within** a single conversational thread with a user. LangGraph manages
short-term memory as a part of your agent's state. State is persisted to a
database using a checkpointer so the thread can be resumed at any time. Short-
term memory updates when the graph is invoked or a step is completed, and the
State is read at the start of each step.

**Long-term memory** is shared **across** conversational threads. It can be
recalled _at any time_ and **in any thread**. Memories are scoped to any
custom namespace, not just within a single thread ID. LangGraph provides
stores (reference doc) to let you save and recall long-term memories.

Both are important to understand and implement for your application.

![](./Memory_files/short-vs-long.png)

## Short-term memory¶

Short-term memory lets your application remember previous interactions within
a single thread or conversation. A thread organizes multiple interactions in a
session, similar to the way email groups messages in a single conversation.

LangGraph manages short-term memory as part of the agent's state, persisted
via thread-scoped checkpoints. This state can normally include the
conversation history along with other stateful data, such as uploaded files,
retrieved documents, or generated artifacts. By storing these in the graph's
state, the bot can access the full context for a given conversation while
maintaining separation between different threads.

Since conversation history is the most common form of representing short-term
memory, in the next section, we will cover techniques for managing
conversation history when the list of messages becomes **long**. If you want
to stick to the high-level concepts, continue on to the long-term memory
section.

### Managing long conversation history¶

Long conversations pose a challenge to today's LLMs. The full history may not
even fit inside an LLM's context window, resulting in an irrecoverable error.
Even _if_ your LLM technically supports the full context length, most LLMs
still perform poorly over long contexts. They get "distracted" by stale or
off-topic content, all while suffering from slower response times and higher
costs.

Managing short-term memory is an exercise of balancing precision & recall with
your application's other performance requirements (latency & cost). As always,
it's important to think critically about how you represent information for
your LLM and to look at your data. We cover a few common techniques for
managing message lists below and hope to provide sufficient context for you to
pick the best tradeoffs for your application:

  * Editing message lists: How to think about trimming and filtering a list of messages before passing to language model.
  * Summarizing past conversations: A common technique to use when you don't just want to filter the list of messages.

### Editing message lists¶

Chat models accept context using messages, which include developer provided
instructions (a system message) and user inputs (human messages). In chat
applications, messages alternate between human inputs and model responses,
resulting in a list of messages that grows longer over time. Because context
windows are limited and token-rich message lists can be costly, many
applications can benefit from using techniques to manually remove or forget
stale information.

![](./Memory_files/filter.png)

The most direct approach is to remove old messages from a list (similar to a
least-recently used cache).

The typical technique for deleting content from a list in LangGraph is to
return an update from a node telling the system to delete some portion of the
list. You get to define what this update looks like, but a common approach
would be to let you return an object or dictionary specifying which values to
retain.

    
    
    def manage_list(existing: list, updates: Union[list, dict]):
        if isinstance(updates, list):
            # Normal case, add to the history
            return existing + updates
        elif isinstance(updates, dict) and updates["type"] == "keep":
            # You get to decide what this looks like.
            # For example, you could simplify and just accept a string "DELETE"
            # and clear the entire list.
            return existing[updates["from"]:updates["to"]]
        # etc. We define how to interpret updates
    
    class State(TypedDict):
        my_list: Annotated[list, manage_list]
    
    def my_node(state: State):
        return {
            # We return an update for the field "my_list" saying to
            # keep only values from index -5 to the end (deleting the rest)
            "my_list": {"type": "keep", "from": -5, "to": None}
        }
    

LangGraph will call the `manage_list` "reducer" function any time an update is
returned under the key "my_list". Within that function, we define what types
of updates to accept. Typically, messages will be added to the existing list
(the conversation will grow); however, we've also added support to accept a
dictionary that lets you "keep" certain parts of the state. This lets you
programmatically drop old message context.

Another common approach is to let you return a list of "remove" objects that
specify the IDs of all messages to delete. If you're using the LangChain
messages and the `add_messages` reducer (or `MessagesState`, which uses the
same underlying functionality) in LangGraph, you can do this using a
`RemoveMessage`.

    
    
    from langchain_core.messages import RemoveMessage, AIMessage
    from langgraph.graph import add_messages
    # ... other imports
    
    class State(TypedDict):
        # add_messages will default to upserting messages by ID to the existing list
        # if a RemoveMessage is returned, it will delete the message in the list by ID
        messages: Annotated[list, add_messages]
    
    def my_node_1(state: State):
        # Add an AI message to the `messages` list in the state
        return {"messages": [AIMessage(content="Hi")]}
    
    def my_node_2(state: State):
        # Delete all but the last 2 messages from the `messages` list in the state
        delete_messages = [RemoveMessage(id=m.id) for m in state['messages'][:-2]]
        return {"messages": delete_messages}
    

API Reference: RemoveMessage | AIMessage | add_messages

In the example above, the `add_messages` reducer allows us to append new
messages to the `messages` state key as shown in `my_node_1`. When it sees a
`RemoveMessage`, it will delete the message with that ID from the list (and
the RemoveMessage will then be discarded). For more information on LangChain-
specific message handling, check out this how-to on using `RemoveMessage` .

See this how-to guide and module 2 from our LangChain Academy course for
example usage.

### Summarizing past conversations¶

The problem with trimming or removing messages, as shown above, is that we may
lose information from culling of the message queue. Because of this, some
applications benefit from a more sophisticated approach of summarizing the
message history using a chat model.

![](./Memory_files/summary.png)

Simple prompting and orchestration logic can be used to achieve this. As an
example, in LangGraph we can extend the MessagesState to include a `summary`
key.

    
    
    from langgraph.graph import MessagesState
    class State(MessagesState):
        summary: str
    

Then, we can generate a summary of the chat history, using any existing
summary as context for the next summary. This `summarize_conversation` node
can be called after some number of messages have accumulated in the `messages`
state key.

    
    
    def summarize_conversation(state: State):
    
        # First, we get any existing summary
        summary = state.get("summary", "")
    
        # Create our summarization prompt
        if summary:
    
            # A summary already exists
            summary_message = (
                f"This is a summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
    
        else:
            summary_message = "Create a summary of the conversation above:"
    
        # Add prompt to our history
        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = model.invoke(messages)
    
        # Delete all but the 2 most recent messages
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        return {"summary": response.content, "messages": delete_messages}
    

See this how-to here and module 2 from our LangChain Academy course for
example usage.

### Knowing **when** to remove messages¶

Most LLMs have a maximum supported context window (denominated in tokens). A
simple way to decide when to truncate messages is to count the tokens in the
message history and truncate whenever it approaches that limit. Naive
truncation is straightforward to implement on your own, though there are a few
"gotchas". Some model APIs further restrict the sequence of message types
(must start with human message, cannot have consecutive messages of the same
type, etc.). If you're using LangChain, you can use the `trim_messages`
utility and specify the number of tokens to keep from the list, as well as the
`strategy` (e.g., keep the last `max_tokens`) to use for handling the
boundary.

Below is an example.

    
    
    from langchain_core.messages import trim_messages
    trim_messages(
        messages,
        # Keep the last <= n_count tokens of the messages.
        strategy="last",
        # Remember to adjust based on your model
        # or else pass a custom token_encoder
        token_counter=ChatOpenAI(model="gpt-4"),
        # Remember to adjust based on the desired conversation
        # length
        max_tokens=45,
        # Most chat models expect that chat history starts with either:
        # (1) a HumanMessage or
        # (2) a SystemMessage followed by a HumanMessage
        start_on="human",
        # Most chat models expect that chat history ends with either:
        # (1) a HumanMessage or
        # (2) a ToolMessage
        end_on=("human", "tool"),
        # Usually, we want to keep the SystemMessage
        # if it's present in the original history.
        # The SystemMessage has special instructions for the model.
        include_system=True,
    )
    

API Reference: trim_messages

## Long-term memory¶

Long-term memory in LangGraph allows systems to retain information across
different conversations or sessions. Unlike short-term memory, which is
**thread-scoped** , long-term memory is saved within custom "namespaces."

### Storing memories¶

LangGraph stores long-term memories as JSON documents in a store (reference
doc). Each memory is organized under a custom `namespace` (similar to a
folder) and a distinct `key` (like a filename). Namespaces often include user
or org IDs or other labels that makes it easier to organize information. This
structure enables hierarchical organization of memories. Cross-namespace
searching is then supported through content filters. See the example below for
an example.

    
    
    from langgraph.store.memory import InMemoryStore
    
    
    def embed(texts: list[str]) -> list[list[float]]:
        # Replace with an actual embedding function or LangChain embeddings object
        return [[1.0, 2.0] * len(texts)]
    
    
    # InMemoryStore saves data to an in-memory dictionary. Use a DB-backed store in production use.
    store = InMemoryStore(index={"embed": embed, "dims": 2})
    user_id = "my-user"
    application_context = "chitchat"
    namespace = (user_id, application_context)
    store.put(
        namespace,
        "a-memory",
        {
            "rules": [
                "User likes short, direct language",
                "User only speaks English & python",
            ],
            "my-key": "my-value",
        },
    )
    # get the "memory" by ID
    item = store.get(namespace, "a-memory")
    # search for "memories" within this namespace, filtering on content equivalence, sorted by vector similarity
    items = store.search(
        namespace, filter={"my-key": "my-value"}, query="language preferences"
    )
    

### Framework for thinking about long-term memory¶

Long-term memory is a complex challenge without a one-size-fits-all solution.
However, the following questions provide a structure framework to help you
navigate the different techniques:

**What is the type of memory?**

Humans use memories to remember facts, experiences, and rules. AI agents can
use memory in the same ways. For example, AI agents can use memory to remember
specific facts about a user to accomplish a task. We expand on several types
of memories in the section below.

**When do you want to update memories?**

Memory can be updated as part of an agent's application logic (e.g. "on the
hot path"). In this case, the agent typically decides to remember facts before
responding to a user. Alternatively, memory can be updated as a background
task (logic that runs in the background / asynchronously and generates
memories). We explain the tradeoffs between these approaches in the section
below.

## Memory types¶

Different applications require various types of memory. Although the analogy
isn't perfect, examining human memory types can be insightful. Some research
(e.g., the CoALA paper) have even mapped these human memory types to those
used in AI agents.

Memory Type | What is Stored | Human Example | Agent Example  
---|---|---|---  
Semantic | Facts | Things I learned in school | Facts about a user  
Episodic | Experiences | Things I did | Past agent actions  
Procedural | Instructions | Instincts or motor skills | Agent system prompt  
  
### Semantic Memory¶

Semantic memory, both in humans and AI agents, involves the retention of
specific facts and concepts. In humans, it can include information learned in
school and the understanding of concepts and their relationships. For AI
agents, semantic memory is often used to personalize applications by
remembering facts or concepts from past interactions.

> Note: Not to be confused with "semantic search" which is a technique for
> finding similar content using "meaning" (usually as embeddings). Semantic
> memory is a term from psychology, referring to storing facts and knowledge,
> while semantic search is a method for retrieving information based on
> meaning rather than exact matches.

#### Profile¶

Semantic memories can be managed in different ways. For example, memories can
be a single, continuously updated "profile" of well-scoped and specific
information about a user, organization, or other entity (including the agent
itself). A profile is generally just a JSON document with various key-value
pairs you've selected to represent your domain.

When remembering a profile, you will want to make sure that you are
**updating** the profile each time. As a result, you will want to pass in the
previous profile and ask the model to generate a new profile (or some JSON
patch to apply to the old profile). This can be become error-prone as the
profile gets larger, and may benefit from splitting a profile into multiple
documents or **strict** decoding when generating documents to ensure the
memory schemas remains valid.

![](./Memory_files/update-profile.png)

#### Collection¶

Alternatively, memories can be a collection of documents that are continuously
updated and extended over time. Each individual memory can be more narrowly
scoped and easier to generate, which means that you're less likely to **lose**
information over time. It's easier for an LLM to generate _new_ objects for
new information than reconcile new information with an existing profile. As a
result, a document collection tends to lead to higher recall downstream.

However, this shifts some complexity memory updating. The model must now
_delete_ or _update_ existing items in the list, which can be tricky. In
addition, some models may default to over-inserting and others may default to
over-updating. See the Trustcall package for one way to manage this and
consider evaluation (e.g., with a tool like LangSmith) to help you tune the
behavior.

Working with document collections also shifts complexity to memory **search**
over the list. The `Store` currently supports both semantic search and
filtering by content.

Finally, using a collection of memories can make it challenging to provide
comprehensive context to the model. While individual memories may follow a
specific schema, this structure might not capture the full context or
relationships between memories. As a result, when using these memories to
generate responses, the model may lack important contextual information that
would be more readily available in a unified profile approach.

![](./Memory_files/update-list.png)

Regardless of memory management approach, the central point is that the agent
will use the semantic memories to ground its responses, which often leads to
more personalized and relevant interactions.

### Episodic Memory¶

Episodic memory, in both humans and AI agents, involves recalling past events
or actions. The CoALA paper frames this well: facts can be written to semantic
memory, whereas _experiences_ can be written to episodic memory. For AI
agents, episodic memory is often used to help an agent remember how to
accomplish a task.

In practice, episodic memories are often implemented through few-shot example
prompting, where agents learn from past sequences to perform tasks correctly.
Sometimes it's easier to "show" than "tell" and LLMs learn well from examples.
Few-shot learning lets you "program" your LLM by updating the prompt with
input-output examples to illustrate the intended behavior. While various best-
practices can be used to generate few-shot examples, often the challenge lies
in selecting the most relevant examples based on user input.

Note that the memory store is just one way to store data as few-shot examples.
If you want to have more developer involvement, or tie few-shots more closely
to your evaluation harness, you can also use a LangSmith Dataset to store your
data. Then dynamic few-shot example selectors can be used out-of-the box to
achieve this same goal. LangSmith will index the dataset for you and enable
retrieval of few shot examples that are most relevant to the user input based
upon keyword similarity (using a BM25-like algorithm for keyword based
similarity).

See this how-to video for example usage of dynamic few-shot example selection
in LangSmith. Also, see this blog post showcasing few-shot prompting to
improve tool calling performance and this blog post using few-shot example to
align an LLMs to human preferences.

### Procedural Memory¶

Procedural memory, in both humans and AI agents, involves remembering the
rules used to perform tasks. In humans, procedural memory is like the
internalized knowledge of how to perform tasks, such as riding a bike via
basic motor skills and balance. Episodic memory, on the other hand, involves
recalling specific experiences, such as the first time you successfully rode a
bike without training wheels or a memorable bike ride through a scenic route.
For AI agents, procedural memory is a combination of model weights, agent
code, and agent's prompt that collectively determine the agent's
functionality.

In practice, it is fairly uncommon for agents to modify their model weights or
rewrite their code. However, it is more common for agents to modify their own
prompts.

One effective approach to refining an agent's instructions is through
"Reflection" or meta-prompting. This involves prompting the agent with its
current instructions (e.g., the system prompt) along with recent conversations
or explicit user feedback. The agent then refines its own instructions based
on this input. This method is particularly useful for tasks where instructions
are challenging to specify upfront, as it allows the agent to learn and adapt
from its interactions.

For example, we built a Tweet generator using external feedback and prompt re-
writing to produce high-quality paper summaries for Twitter. In this case, the
specific summarization prompt was difficult to specify _a priori_ , but it was
fairly easy for a user to critique the generated Tweets and provide feedback
on how to improve the summarization process.

The below pseudo-code shows how you might implement this with the LangGraph
memory store, using the store to save a prompt, the `update_instructions` node
to get the current prompt (as well as feedback from the conversation with the
user captured in `state["messages"]`), update the prompt, and save the new
prompt back to the store. Then, the `call_model` get the updated prompt from
the store and uses it to generate a response.

    
    
    # Node that *uses* the instructions
    def call_model(state: State, store: BaseStore):
        namespace = ("agent_instructions", )
        instructions = store.get(namespace, key="agent_a")[0]
        # Application logic
        prompt = prompt_template.format(instructions=instructions.value["instructions"])
        ...
    
    # Node that updates instructions
    def update_instructions(state: State, store: BaseStore):
        namespace = ("instructions",)
        current_instructions = store.search(namespace)[0]
        # Memory logic
        prompt = prompt_template.format(instructions=instructions.value["instructions"], conversation=state["messages"])
        output = llm.invoke(prompt)
        new_instructions = output['new_instructions']
        store.put(("agent_instructions",), "agent_a", {"instructions": new_instructions})
        ...
    

![](./Memory_files/update-instructions.png)

## Writing memories¶

While humans often form long-term memories during sleep, AI agents need a
different approach. When and how should agents create new memories? There are
at least two primary methods for agents to write memories: "on the hot path"
and "in the background".

![](./Memory_files/hot_path_vs_background.png)

### Writing memories in the hot path¶

Creating memories during runtime offers both advantages and challenges. On the
positive side, this approach allows for real-time updates, making new memories
immediately available for use in subsequent interactions. It also enables
transparency, as users can be notified when memories are created and stored.

However, this method also presents challenges. It may increase complexity if
the agent requires a new tool to decide what to commit to memory. In addition,
the process of reasoning about what to save to memory can impact agent
latency. Finally, the agent must multitask between memory creation and its
other responsibilities, potentially affecting the quantity and quality of
memories created.

As an example, ChatGPT uses a save_memories tool to upsert memories as content
strings, deciding whether and how to use this tool with each user message. See
our memory-agent template as an reference implementation.

### Writing memories in the background¶

Creating memories as a separate background task offers several advantages. It
eliminates latency in the primary application, separates application logic
from memory management, and allows for more focused task completion by the
agent. This approach also provides flexibility in timing memory creation to
avoid redundant work.

However, this method has its own challenges. Determining the frequency of
memory writing becomes crucial, as infrequent updates may leave other threads
without new context. Deciding when to trigger memory formation is also
important. Common strategies include scheduling after a set time period (with
rescheduling if new events occur), using a cron schedule, or allowing manual
triggers by users or the application logic.

See our memory-service template as an reference implementation.

## Comments

Back to top

Previous

Persistence

Next

Streaming

Made with  Material for MkDocs Insiders



Skip to content

To learn more about LangGraph, check out our first LangChain Academy course,
_Introduction to LangGraph_ , available for free here.

![logo](./Multi-agent Systems_files/wordmark_dark.svg) ![logo](./Multi-agent
Systems_files/wordmark_light.svg)

Multi-agent Systems

Type to start searching

GitHub

  * 0.2.62
  * 7.9k
  * 1.3k

  * Home 
  * Tutorials 
  * How-to Guides 
  * Conceptual Guides 
  * Reference 

![logo](./Multi-agent Systems_files/wordmark_dark.svg) ![logo](./Multi-agent
Systems_files/wordmark_light.svg)

GitHub

  * 0.2.62
  * 7.9k
  * 1.3k

  * Home 
  * Tutorials 
  * How-to Guides 
  * Conceptual Guides 

Conceptual Guides

    * LangGraph  LangGraph 
      * LangGraph 
      * Why LangGraph? 
      * LangGraph Glossary 
      * Agent architectures 
      * Multi-agent Systems  Multi-agent Systems  Table of contents 
        * Multi-agent architectures 
          * Handoffs 
            * Handoffs as tools 
          * Network 
          * Supervisor 
          * Supervisor (tool-calling) 
          * Hierarchical 
          * Custom multi-agent workflow 
        * Communication between agents 
          * Graph state vs tool calls 
            * Graph state 
          * Different state schemas 
          * Shared message list 
            * Share full history 
            * Share final result 
      * Human-in-the-loop 
      * Persistence 
      * Memory 
      * Streaming 
      * FAQ 
    * LangGraph Platform  LangGraph Platform 
      * LangGraph Platform 
      * High Level 
      * Components 
      * LangGraph Server 
      * Deployment Options 
  * Reference 

Table of contents

  * Multi-agent architectures 
    * Handoffs 
      * Handoffs as tools 
    * Network 
    * Supervisor 
    * Supervisor (tool-calling) 
    * Hierarchical 
    * Custom multi-agent workflow 
  * Communication between agents 
    * Graph state vs tool calls 
      * Graph state 
    * Different state schemas 
    * Shared message list 
      * Share full history 
      * Share final result 

  1. Home 
  2. Conceptual Guides 
  3. LangGraph 

# Multi-agent Systems¶

An agent is _a system that uses an LLM to decide the control flow of an
application_. As you develop these systems, they might grow more complex over
time, making them harder to manage and scale. For example, you might run into
the following problems:

  * agent has too many tools at its disposal and makes poor decisions about which tool to call next
  * context grows too complex for a single agent to keep track of
  * there is a need for multiple specialization areas in the system (e.g. planner, researcher, math expert, etc.)

To tackle these, you might consider breaking your application into multiple
smaller, independent agents and composing them into a **multi-agent system**.
These independent agents can be as simple as a prompt and an LLM call, or as
complex as a ReAct agent (and more!).

The primary benefits of using multi-agent systems are:

  * **Modularity** : Separate agents make it easier to develop, test, and maintain agentic systems.
  * **Specialization** : You can create expert agents focused on specific domains, which helps with the overall system performance.
  * **Control** : You can explicitly control how agents communicate (as opposed to relying on function calling).

## Multi-agent architectures¶

![](./Multi-agent Systems_files/architectures.png)

There are several ways to connect agents in a multi-agent system:

  * **Network** : each agent can communicate with every other agent. Any agent can decide which other agent to call next.
  * **Supervisor** : each agent communicates with a single supervisor agent. Supervisor agent makes decisions on which agent should be called next.
  * **Supervisor (tool-calling)** : this is a special case of supervisor architecture. Individual agents can be represented as tools. In this case, a supervisor agent uses a tool-calling LLM to decide which of the agent tools to call, as well as the arguments to pass to those agents.
  * **Hierarchical** : you can define a multi-agent system with a supervisor of supervisors. This is a generalization of the supervisor architecture and allows for more complex control flows.
  * **Custom multi-agent workflow** : each agent communicates with only a subset of agents. Parts of the flow are deterministic, and only some agents can decide which other agents to call next.

### Handoffs¶

In multi-agent architectures, agents can be represented as graph nodes. Each
agent node executes its step(s) and decides whether to finish execution or
route to another agent, including potentially routing to itself (e.g., running
in a loop). A common pattern in multi-agent interactions is handoffs, where
one agent hands off control to another. Handoffs allow you to specify:

  * **destination** : target agent to navigate to (e.g., name of the node to go to)
  * **payload** : information to pass to that agent (e.g., state update)

To implement handoffs in LangGraph, agent nodes can return `Command` object
that allows you to combine both control flow and state updates:

    
    
    def agent(state) -> Command[Literal["agent", "another_agent"]]:
        # the condition for routing/halting can be anything, e.g. LLM tool call / structured output, etc.
        goto = get_next_agent(...)  # 'agent' / 'another_agent'
        return Command(
            # Specify which agent to call next
            goto=goto,
            # Update the graph state
            update={"my_state_key": "my_state_value"}
        )
    

In a more complex scenario where each agent node is itself a graph (i.e., a
subgraph), a node in one of the agent subgraphs might want to navigate to a
different agent. For example, if you have two agents, `alice` and `bob`
(subgraph nodes in a parent graph), and `alice` needs to navigate to `bob`,
you can set `graph=Command.PARENT` in the `Command` object:

    
    
    def some_node_inside_alice(state)
        return Command(
            goto="bob",
            update={"my_state_key": "my_state_value"},
            # specify which graph to navigate to (defaults to the current graph)
            graph=Command.PARENT,
        )
    

Note

If you need to support visualization for subgraphs communicating using
`Command(graph=Command.PARENT)` you would need to wrap them in a node function
with `Command` annotation, e.g. instead of this:

    
    
    builder.add_node(alice)
    

you would need to do this:

    
    
    def call_alice(state) -> Command[Literal["bob"]]:
        return alice.invoke(state)
    
    builder.add_node("alice", call_alice)
    

#### Handoffs as tools¶

One of the most common agent types is a ReAct-style tool-calling agents. For
those types of agents, a common pattern is wrapping a handoff in a tool call,
e.g.:

    
    
    def transfer_to_bob(state):
        """Transfer to bob."""
        return Command(
            goto="bob",
            update={"my_state_key": "my_state_value"},
            graph=Command.PARENT,
        )
    

This is a special case of updating the graph state from tools where in
addition the state update, the control flow is included as well.

Important

If you want to use tools that return `Command`, you can either use prebuilt
`create_react_agent` / `ToolNode` components, or implement your own tool-
executing node that collects `Command` objects returned by the tools and
returns a list of them, e.g.:

    
    
    def call_tools(state):
        ...
        commands = [tools_by_name[tool_call["name"]].invoke(tool_call) for tool_call in tool_calls]
        return commands
    

Let's now take a closer look at the different multi-agent architectures.

### Network¶

In this architecture, agents are defined as graph nodes. Each agent can
communicate with every other agent (many-to-many connections) and can decide
which agent to call next. This architecture is good for problems that do not
have a clear hierarchy of agents or a specific sequence in which agents should
be called.

    
    
    from typing import Literal
    from langchain_openai import ChatOpenAI
    from langgraph.graph import StateGraph, MessagesState, START
    
    model = ChatOpenAI()
    
    def agent_1(state: MessagesState) -> Command[Literal["agent_2", "agent_3", END]]:
        # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
        # to determine which agent to call next. a common pattern is to call the model
        # with a structured output (e.g. force it to return an output with a "next_agent" field)
        response = model.invoke(...)
        # route to one of the agents or exit based on the LLM's decision
        # if the LLM returns "__end__", the graph will finish execution
        return Command(
            goto=response["next_agent"],
            update={"messages": [response["content"]]},
        )
    
    def agent_2(state: MessagesState) -> Command[Literal["agent_1", "agent_3", END]]:
        response = model.invoke(...)
        return Command(
            goto=response["next_agent"],
            update={"messages": [response["content"]]},
        )
    
    def agent_3(state: MessagesState) -> Command[Literal["agent_1", "agent_2", END]]:
        ...
        return Command(
            goto=response["next_agent"],
            update={"messages": [response["content"]]},
        )
    
    builder = StateGraph(MessagesState)
    builder.add_node(agent_1)
    builder.add_node(agent_2)
    builder.add_node(agent_3)
    
    builder.add_edge(START, "agent_1")
    network = builder.compile()
    

API Reference: ChatOpenAI | StateGraph | START

### Supervisor¶

In this architecture, we define agents as nodes and add a supervisor node
(LLM) that decides which agent nodes should be called next. We use `Command`
to route execution to the appropriate agent node based on supervisor's
decision. This architecture also lends itself well to running multiple agents
in parallel or using map-reduce pattern.

    
    
    from typing import Literal
    from langchain_openai import ChatOpenAI
    from langgraph.graph import StateGraph, MessagesState, START, END
    
    model = ChatOpenAI()
    
    def supervisor(state: MessagesState) -> Command[Literal["agent_1", "agent_2", END]]:
        # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
        # to determine which agent to call next. a common pattern is to call the model
        # with a structured output (e.g. force it to return an output with a "next_agent" field)
        response = model.invoke(...)
        # route to one of the agents or exit based on the supervisor's decision
        # if the supervisor returns "__end__", the graph will finish execution
        return Command(goto=response["next_agent"])
    
    def agent_1(state: MessagesState) -> Command[Literal["supervisor"]]:
        # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
        # and add any additional logic (different models, custom prompts, structured output, etc.)
        response = model.invoke(...)
        return Command(
            goto="supervisor",
            update={"messages": [response]},
        )
    
    def agent_2(state: MessagesState) -> Command[Literal["supervisor"]]:
        response = model.invoke(...)
        return Command(
            goto="supervisor",
            update={"messages": [response]},
        )
    
    builder = StateGraph(MessagesState)
    builder.add_node(supervisor)
    builder.add_node(agent_1)
    builder.add_node(agent_2)
    
    builder.add_edge(START, "supervisor")
    
    supervisor = builder.compile()
    

API Reference: ChatOpenAI | StateGraph | START | END

Check out this tutorial for an example of supervisor multi-agent architecture.

### Supervisor (tool-calling)¶

In this variant of the supervisor architecture, we define individual agents as
**tools** and use a tool-calling LLM in the supervisor node. This can be
implemented as a ReAct-style agent with two nodes — an LLM node (supervisor)
and a tool-calling node that executes tools (agents in this case).

    
    
    from typing import Annotated
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import InjectedState, create_react_agent
    
    model = ChatOpenAI()
    
    # this is the agent function that will be called as tool
    # notice that you can pass the state to the tool via InjectedState annotation
    def agent_1(state: Annotated[dict, InjectedState]):
        # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
        # and add any additional logic (different models, custom prompts, structured output, etc.)
        response = model.invoke(...)
        # return the LLM response as a string (expected tool response format)
        # this will be automatically turned to ToolMessage
        # by the prebuilt create_react_agent (supervisor)
        return response.content
    
    def agent_2(state: Annotated[dict, InjectedState]):
        response = model.invoke(...)
        return response.content
    
    tools = [agent_1, agent_2]
    # the simplest way to build a supervisor w/ tool-calling is to use prebuilt ReAct agent graph
    # that consists of a tool-calling LLM node (i.e. supervisor) and a tool-executing node
    supervisor = create_react_agent(model, tools)
    

API Reference: ChatOpenAI | InjectedState | create_react_agent

### Hierarchical¶

As you add more agents to your system, it might become too hard for the
supervisor to manage all of them. The supervisor might start making poor
decisions about which agent to call next, the context might become too complex
for a single supervisor to keep track of. In other words, you end up with the
same problems that motivated the multi-agent architecture in the first place.

To address this, you can design your system _hierarchically_. For example, you
can create separate, specialized teams of agents managed by individual
supervisors, and a top-level supervisor to manage the teams.

    
    
    from typing import Literal
    from langchain_openai import ChatOpenAI
    from langgraph.graph import StateGraph, MessagesState, START, END
    
    model = ChatOpenAI()
    
    # define team 1 (same as the single supervisor example above)
    
    def team_1_supervisor(state: MessagesState) -> Command[Literal["team_1_agent_1", "team_1_agent_2", END]]:
        response = model.invoke(...)
        return Command(goto=response["next_agent"])
    
    def team_1_agent_1(state: MessagesState) -> Command[Literal["team_1_supervisor"]]:
        response = model.invoke(...)
        return Command(goto="team_1_supervisor", update={"messages": [response]})
    
    def team_1_agent_2(state: MessagesState) -> Command[Literal["team_1_supervisor"]]:
        response = model.invoke(...)
        return Command(goto="team_1_supervisor", update={"messages": [response]})
    
    team_1_builder = StateGraph(Team1State)
    team_1_builder.add_node(team_1_supervisor)
    team_1_builder.add_node(team_1_agent_1)
    team_1_builder.add_node(team_1_agent_2)
    team_1_builder.add_edge(START, "team_1_supervisor")
    team_1_graph = team_1_builder.compile()
    
    # define team 2 (same as the single supervisor example above)
    class Team2State(MessagesState):
        next: Literal["team_2_agent_1", "team_2_agent_2", "__end__"]
    
    def team_2_supervisor(state: Team2State):
        ...
    
    def team_2_agent_1(state: Team2State):
        ...
    
    def team_2_agent_2(state: Team2State):
        ...
    
    team_2_builder = StateGraph(Team2State)
    ...
    team_2_graph = team_2_builder.compile()
    
    
    # define top-level supervisor
    
    builder = StateGraph(MessagesState)
    def top_level_supervisor(state: MessagesState):
        # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
        # to determine which team to call next. a common pattern is to call the model
        # with a structured output (e.g. force it to return an output with a "next_team" field)
        response = model.invoke(...)
        # route to one of the teams or exit based on the supervisor's decision
        # if the supervisor returns "__end__", the graph will finish execution
        return Command(goto=response["next_team"])
    
    builder = StateGraph(MessagesState)
    builder.add_node(top_level_supervisor)
    builder.add_node(team_1_graph)
    builder.add_node(team_2_graph)
    
    builder.add_edge(START, "top_level_supervisor")
    graph = builder.compile()
    

API Reference: ChatOpenAI | StateGraph | START | END

### Custom multi-agent workflow¶

In this architecture we add individual agents as graph nodes and define the
order in which agents are called ahead of time, in a custom workflow. In
LangGraph the workflow can be defined in two ways:

  * **Explicit control flow (normal edges)** : LangGraph allows you to explicitly define the control flow of your application (i.e. the sequence of how agents communicate) explicitly, via normal graph edges. This is the most deterministic variant of this architecture above — we always know which agent will be called next ahead of time.

  * **Dynamic control flow (Command)** : in LangGraph you can allow LLMs to decide parts of your application control flow. This can be achieved by using `Command`. A special case of this is a supervisor tool-calling architecture. In that case, the tool-calling LLM powering the supervisor agent will make decisions about the order in which the tools (agents) are being called.

    
    
    from langchain_openai import ChatOpenAI
    from langgraph.graph import StateGraph, MessagesState, START
    
    model = ChatOpenAI()
    
    def agent_1(state: MessagesState):
        response = model.invoke(...)
        return {"messages": [response]}
    
    def agent_2(state: MessagesState):
        response = model.invoke(...)
        return {"messages": [response]}
    
    builder = StateGraph(MessagesState)
    builder.add_node(agent_1)
    builder.add_node(agent_2)
    # define the flow explicitly
    builder.add_edge(START, "agent_1")
    builder.add_edge("agent_1", "agent_2")
    

API Reference: ChatOpenAI | StateGraph | START

## Communication between agents¶

The most important thing when building multi-agent systems is figuring out how
the agents communicate. There are few different considerations:

  * Do agents communicate via **via graph state or via tool calls**?
  * What if two agents have **different state schemas**?
  * How to communicate over a **shared message list**?

### Graph state vs tool calls¶

What is the "payload" that is being passed around between agents? In most of
the architectures discussed above the agents communicate via the graph state.
In the case of the supervisor with tool-calling, the payloads are tool call
arguments.

![](./Multi-agent Systems_files/request.png)

#### Graph state¶

To communicate via graph state, individual agents need to be defined as graph
nodes. These can be added as functions or as entire subgraphs. At each step of
the graph execution, agent node receives the current state of the graph,
executes the agent code and then passes the updated state to the next nodes.

Typically agent nodes share a single state schema. However, you might want to
design agent nodes with different state schemas.

### Different state schemas¶

An agent might need to have a different state schema from the rest of the
agents. For example, a search agent might only need to keep track of queries
and retrieved documents. There are two ways to achieve this in LangGraph:

  * Define subgraph agents with a separate state schema. If there are no shared state keys (channels) between the subgraph and the parent graph, it’s important to add input / output transformations so that the parent graph knows how to communicate with the subgraphs.
  * Define agent node functions with a private input state schema that is distinct from the overall graph state schema. This allows passing information that is only needed for executing that particular agent.

### Shared message list¶

The most common way for the agents to communicate is via a shared state
channel, typically a list of messages. This assumes that there is always at
least a single channel (key) in the state that is shared by the agents. When
communicating via a shared message list there is an additional consideration:
should the agents share the full history of their thought process or only the
final result?

![](./Multi-agent Systems_files/response.png)

#### Share full history¶

Agents can **share the full history** of their thought process (i.e.
"scratchpad") with all other agents. This "scratchpad" would typically look
like a list of messages. The benefit of sharing full thought process is that
it might help other agents make better decisions and improve reasoning ability
for the system as a whole. The downside is that as the number of agents and
their complexity grows, the "scratchpad" will grow quickly and might require
additional strategies for memory management.

#### Share final result¶

Agents can have their own private "scratchpad" and only **share the final
result** with the rest of the agents. This approach might work better for
systems with many agents or agents that are more complex. In this case, you
would need to define agents with different state schemas

For agents called as tools, the supervisor determines the inputs based on the
tool schema. Additionally, LangGraph allows passing state to individual tools
at runtime, so subordinate agents can access parent state, if needed.

## Comments

Back to top

Previous

Agent architectures

Next

Human-in-the-loop

Made with  Material for MkDocs Insiders



Skip to content

To learn more about LangGraph, check out our first LangChain Academy course,
_Introduction to LangGraph_ , available for free here.

![logo](./Persistence_files/wordmark_dark.svg)
![logo](./Persistence_files/wordmark_light.svg)

Persistence

Type to start searching

GitHub

  * 0.2.62
  * 7.9k
  * 1.3k

  * Home 
  * Tutorials 
  * How-to Guides 
  * Conceptual Guides 
  * Reference 

![logo](./Persistence_files/wordmark_dark.svg)
![logo](./Persistence_files/wordmark_light.svg)

GitHub

  * 0.2.62
  * 7.9k
  * 1.3k

  * Home 
  * Tutorials 
  * How-to Guides 
  * Conceptual Guides 

Conceptual Guides

    * LangGraph  LangGraph 
      * LangGraph 
      * Why LangGraph? 
      * LangGraph Glossary 
      * Agent architectures 
      * Multi-agent Systems 
      * Human-in-the-loop 
      * Persistence  Persistence  Table of contents 
        * Threads 
        * Checkpoints 
          * Get state 
          * Get state history 
          * Replay 
          * Update state 
            * config 
            * values 
            * as_node 
        * Memory Store 
          * Basic Usage 
          * Semantic Search 
          * Using in LangGraph 
        * Checkpointer libraries 
          * Checkpointer interface 
          * Serializer 
        * Capabilities 
          * Human-in-the-loop 
          * Memory 
          * Time Travel 
          * Fault-tolerance 
            * Pending writes 
      * Memory 
      * Streaming 
      * FAQ 
    * LangGraph Platform  LangGraph Platform 
      * LangGraph Platform 
      * High Level 
      * Components 
      * LangGraph Server 
      * Deployment Options 
  * Reference 

Table of contents

  * Threads 
  * Checkpoints 
    * Get state 
    * Get state history 
    * Replay 
    * Update state 
      * config 
      * values 
      * as_node 
  * Memory Store 
    * Basic Usage 
    * Semantic Search 
    * Using in LangGraph 
  * Checkpointer libraries 
    * Checkpointer interface 
    * Serializer 
  * Capabilities 
    * Human-in-the-loop 
    * Memory 
    * Time Travel 
    * Fault-tolerance 
      * Pending writes 

  1. Home 
  2. Conceptual Guides 
  3. LangGraph 

# Persistence¶

LangGraph has a built-in persistence layer, implemented through checkpointers.
When you compile graph with a checkpointer, the checkpointer saves a
`checkpoint` of the graph state at every super-step. Those checkpoints are
saved to a `thread`, which can be accessed after graph execution. Because
`threads` allow access to graph's state after execution, several powerful
capabilities including human-in-the-loop, memory, time travel, and fault-
tolerance are all possible. See this how-to guide for an end-to-end example on
how to add and use checkpointers with your graph. Below, we'll discuss each of
these concepts in more detail.

![Checkpoints](./Persistence_files/checkpoints.jpg)

## Threads¶

A thread is a unique ID or thread identifier assigned to each checkpoint saved
by a checkpointer. When invoking graph with a checkpointer, you **must**
specify a `thread_id` as part of the `configurable` portion of the config:

    
    
    {"configurable": {"thread_id": "1"}}
    

## Checkpoints¶

Checkpoint is a snapshot of the graph state saved at each super-step and is
represented by `StateSnapshot` object with the following key properties:

  * `config`: Config associated with this checkpoint. 
  * `metadata`: Metadata associated with this checkpoint.
  * `values`: Values of the state channels at this point in time.
  * `next` A tuple of the node names to execute next in the graph.
  * `tasks`: A tuple of `PregelTask` objects that contain information about next tasks to be executed. If the step was previously attempted, it will include error information. If a graph was interrupted dynamically from within a node, tasks will contain additional data associated with interrupts.

Let's see what checkpoints are saved when a simple graph is invoked as
follows:

    
    
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver
    from typing import Annotated
    from typing_extensions import TypedDict
    from operator import add
    
    class State(TypedDict):
        foo: int
        bar: Annotated[list[str], add]
    
    def node_a(state: State):
        return {"foo": "a", "bar": ["a"]}
    
    def node_b(state: State):
        return {"foo": "b", "bar": ["b"]}
    
    
    workflow = StateGraph(State)
    workflow.add_node(node_a)
    workflow.add_node(node_b)
    workflow.add_edge(START, "node_a")
    workflow.add_edge("node_a", "node_b")
    workflow.add_edge("node_b", END)
    
    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"foo": ""}, config)
    

API Reference: StateGraph | START | END | MemorySaver

After we run the graph, we expect to see exactly 4 checkpoints:

  * empty checkpoint with `START` as the next node to be executed
  * checkpoint with the user input `{'foo': '', 'bar': []}` and `node_a` as the next node to be executed
  * checkpoint with the outputs of `node_a` `{'foo': 'a', 'bar': ['a']}` and `node_b` as the next node to be executed
  * checkpoint with the outputs of `node_b` `{'foo': 'b', 'bar': ['a', 'b']}` and no next nodes to be executed

Note that we `bar` channel values contain outputs from both nodes as we have a
reducer for `bar` channel.

### Get state¶

When interacting with the saved graph state, you **must** specify a thread
identifier. You can view the _latest_ state of the graph by calling
`graph.get_state(config)`. This will return a `StateSnapshot` object that
corresponds to the latest checkpoint associated with the thread ID provided in
the config or a checkpoint associated with a checkpoint ID for the thread, if
provided.

    
    
    # get the latest state snapshot
    config = {"configurable": {"thread_id": "1"}}
    graph.get_state(config)
    
    # get a state snapshot for a specific checkpoint_id
    config = {"configurable": {"thread_id": "1", "checkpoint_id": "1ef663ba-28fe-6528-8002-5a559208592c"}}
    graph.get_state(config)
    

In our example, the output of `get_state` will look like this:

    
    
    StateSnapshot(
        values={'foo': 'b', 'bar': ['a', 'b']},
        next=(),
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28fe-6528-8002-5a559208592c'}},
        metadata={'source': 'loop', 'writes': {'node_b': {'foo': 'b', 'bar': ['b']}}, 'step': 2},
        created_at='2024-08-29T19:19:38.821749+00:00',
        parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f9-6ec4-8001-31981c2c39f8'}}, tasks=()
    )
    

### Get state history¶

You can get the full history of the graph execution for a given thread by
calling `graph.get_state_history(config)`. This will return a list of
`StateSnapshot` objects associated with the thread ID provided in the config.
Importantly, the checkpoints will be ordered chronologically with the most
recent checkpoint / `StateSnapshot` being the first in the list.

    
    
    config = {"configurable": {"thread_id": "1"}}
    list(graph.get_state_history(config))
    

In our example, the output of `get_state_history` will look like this:

    
    
    [
        StateSnapshot(
            values={'foo': 'b', 'bar': ['a', 'b']},
            next=(),
            config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28fe-6528-8002-5a559208592c'}},
            metadata={'source': 'loop', 'writes': {'node_b': {'foo': 'b', 'bar': ['b']}}, 'step': 2},
            created_at='2024-08-29T19:19:38.821749+00:00',
            parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f9-6ec4-8001-31981c2c39f8'}},
            tasks=(),
        ),
        StateSnapshot(
            values={'foo': 'a', 'bar': ['a']}, next=('node_b',),
            config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f9-6ec4-8001-31981c2c39f8'}},
            metadata={'source': 'loop', 'writes': {'node_a': {'foo': 'a', 'bar': ['a']}}, 'step': 1},
            created_at='2024-08-29T19:19:38.819946+00:00',
            parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f4-6b4a-8000-ca575a13d36a'}},
            tasks=(PregelTask(id='6fb7314f-f114-5413-a1f3-d37dfe98ff44', name='node_b', error=None, interrupts=()),),
        ),
        StateSnapshot(
            values={'foo': '', 'bar': []},
            next=('node_a',),
            config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f4-6b4a-8000-ca575a13d36a'}},
            metadata={'source': 'loop', 'writes': None, 'step': 0},
            created_at='2024-08-29T19:19:38.817813+00:00',
            parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f0-6c66-bfff-6723431e8481'}},
            tasks=(PregelTask(id='f1b14528-5ee5-579c-949b-23ef9bfbed58', name='node_a', error=None, interrupts=()),),
        ),
        StateSnapshot(
            values={'bar': []},
            next=('__start__',),
            config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f0-6c66-bfff-6723431e8481'}},
            metadata={'source': 'input', 'writes': {'foo': ''}, 'step': -1},
            created_at='2024-08-29T19:19:38.816205+00:00',
            parent_config=None,
            tasks=(PregelTask(id='6d27aa2e-d72b-5504-a36f-8620e54a76dd', name='__start__', error=None, interrupts=()),),
        )
    ]
    

![State](./Persistence_files/get_state.jpg)

### Replay¶

It's also possible to play-back a prior graph execution. If we `invoking` a
graph with a `thread_id` and a `checkpoint_id`, then we will _re-play_ the
graph from a checkpoint that corresponds to the `checkpoint_id`.

  * `thread_id` is simply the ID of a thread. This is always required.
  * `checkpoint_id` This identifier refers to a specific checkpoint within a thread. 

You must pass these when invoking the graph as part of the `configurable`
portion of the config:

    
    
    # {"configurable": {"thread_id": "1"}}  # valid config
    # {"configurable": {"thread_id": "1", "checkpoint_id": "0c62ca34-ac19-445d-bbb0-5b4984975b2a"}}  # also valid config
    
    config = {"configurable": {"thread_id": "1"}}
    graph.invoke(None, config=config)
    

Importantly, LangGraph knows whether a particular checkpoint has been executed
previously. If it has, LangGraph simply _re-plays_ that particular step in the
graph and does not re-execute the step. See this how to guide on time-travel
to learn more about replaying.

![Replay](./Persistence_files/re_play.jpg)

### Update state¶

In addition to re-playing the graph from specific `checkpoints`, we can also
_edit_ the graph state. We do this using `graph.update_state()`. This method
accepts three different arguments:

#### `config`¶

The config should contain `thread_id` specifying which thread to update. When
only the `thread_id` is passed, we update (or fork) the current state.
Optionally, if we include `checkpoint_id` field, then we fork that selected
checkpoint.

#### `values`¶

These are the values that will be used to update the state. Note that this
update is treated exactly as any update from a node is treated. This means
that these values will be passed to the reducer functions, if they are defined
for some of the channels in the graph state. This means that `update_state`
does NOT automatically overwrite the channel values for every channel, but
only for the channels without reducers. Let's walk through an example.

Let's assume you have defined the state of your graph with the following
schema (see full example above):

    
    
    from typing import Annotated
    from typing_extensions import TypedDict
    from operator import add
    
    class State(TypedDict):
        foo: int
        bar: Annotated[list[str], add]
    

Let's now assume the current state of the graph is

    
    
    {"foo": 1, "bar": ["a"]}
    

If you update the state as below:

    
    
    graph.update_state(config, {"foo": 2, "bar": ["b"]})
    

Then the new state of the graph will be:

    
    
    {"foo": 2, "bar": ["a", "b"]}
    

The `foo` key (channel) is completely changed (because there is no reducer
specified for that channel, so `update_state` overwrites it). However, there
is a reducer specified for the `bar` key, and so it appends `"b"` to the state
of `bar`.

#### `as_node`¶

The final thing you can optionally specify when calling `update_state` is
`as_node`. If you provided it, the update will be applied as if it came from
node `as_node`. If `as_node` is not provided, it will be set to the last node
that updated the state, if not ambiguous. The reason this matters is that the
next steps to execute depend on the last node to have given an update, so this
can be used to control which node executes next. See this how to guide on
time-travel to learn more about forking state.

![Update](./Persistence_files/checkpoints_full_story.jpg)

## Memory Store¶

![Model of shared state](./Persistence_files/shared_state.png)

A state schema specifies a set of keys that are populated as a graph is
executed. As discussed above, state can be written by a checkpointer to a
thread at each graph step, enabling state persistence.

But, what if we want to retain some information _across threads_? Consider the
case of a chatbot where we want to retain specific information about the user
across _all_ chat conversations (e.g., threads) with that user!

With checkpointers alone, we cannot share information across threads. This
motivates the need for the `Store` interface. As an illustration, we can
define an `InMemoryStore` to store information about a user across threads. We
simply compile our graph with a checkpointer, as before, and with our new
`in_memory_store` variable.

### Basic Usage¶

First, let's showcase this in isolation without using LangGraph.

    
    
    from langgraph.store.memory import InMemoryStore
    in_memory_store = InMemoryStore()
    

Memories are namespaced by a `tuple`, which in this specific example will be
`(<user_id>, "memories")`. The namespace can be any length and represent
anything, does not have be user specific.

    
    
    user_id = "1"
    namespace_for_memory = (user_id, "memories")
    

We use the `store.put` method to save memories to our namespace in the store.
When we do this, we specify the namespace, as defined above, and a key-value
pair for the memory: the key is simply a unique identifier for the memory
(`memory_id`) and the value (a dictionary) is the memory itself.

    
    
    memory_id = str(uuid.uuid4())
    memory = {"food_preference" : "I like pizza"}
    in_memory_store.put(namespace_for_memory, memory_id, memory)
    

We can read out memories in our namespace using the `store.search` method,
which will return all memories for a given user as a list. The most recent
memory is the last in the list.

    
    
    memories = in_memory_store.search(namespace_for_memory)
    memories[-1].dict()
    {'value': {'food_preference': 'I like pizza'},
     'key': '07e0caf4-1631-47b7-b15f-65515d4c1843',
     'namespace': ['1', 'memories'],
     'created_at': '2024-10-02T17:22:31.590602+00:00',
     'updated_at': '2024-10-02T17:22:31.590605+00:00'}
    

Each memory type is a Python class (`Item`) with certain attributes. We can
access it as a dictionary by converting via `.dict` as above. The attributes
it has are:

  * `value`: The value (itself a dictionary) of this memory
  * `key`: A unique key for this memory in this namespace
  * `namespace`: A list of strings, the namespace of this memory type
  * `created_at`: Timestamp for when this memory was created
  * `updated_at`: Timestamp for when this memory was updated

### Semantic Search¶

Beyond simple retrieval, the store also supports semantic search, allowing you
to find memories based on meaning rather than exact matches. To enable this,
configure the store with an embedding model:

    
    
    from langchain.embeddings import init_embeddings
    
    store = InMemoryStore(
        index={
            "embed": init_embeddings("openai:text-embedding-3-small"),  # Embedding provider
            "dims": 1536,                              # Embedding dimensions
            "fields": ["food_preference", "$"]              # Fields to embed
        }
    )
    

API Reference: init_embeddings

Now when searching, you can use natural language queries to find relevant
memories:

    
    
    # Find memories about food preferences
    # (This can be done after putting memories into the store)
    memories = store.search(
        namespace_for_memory,
        query="What does the user like to eat?",
        limit=3  # Return top 3 matches
    )
    

You can control which parts of your memories get embedded by configuring the
`fields` parameter or by specifying the `index` parameter when storing
memories:

    
    
    # Store with specific fields to embed
    store.put(
        namespace_for_memory,
        str(uuid.uuid4()),
        {
            "food_preference": "I love Italian cuisine",
            "context": "Discussing dinner plans"
        },
        index=["food_preference"]  # Only embed "food_preferences" field
    )
    
    # Store without embedding (still retrievable, but not searchable)
    store.put(
        namespace_for_memory,
        str(uuid.uuid4()),
        {"system_info": "Last updated: 2024-01-01"},
        index=False
    )
    

### Using in LangGraph¶

With this all in place, we use the `in_memory_store` in LangGraph. The
`in_memory_store` works hand-in-hand with the checkpointer: the checkpointer
saves state to threads, as discussed above, and the `in_memory_store` allows
us to store arbitrary information for access _across_ threads. We compile the
graph with both the checkpointer and the `in_memory_store` as follows.

    
    
    from langgraph.checkpoint.memory import MemorySaver
    
    # We need this because we want to enable threads (conversations)
    checkpointer = MemorySaver()
    
    # ... Define the graph ...
    
    # Compile the graph with the checkpointer and store
    graph = graph.compile(checkpointer=checkpointer, store=in_memory_store)
    

API Reference: MemorySaver

We invoke the graph with a `thread_id`, as before, and also with a `user_id`,
which we'll use to namespace our memories to this particular user as we showed
above.

    
    
    # Invoke the graph
    user_id = "1"
    config = {"configurable": {"thread_id": "1", "user_id": user_id}}
    
    # First let's just say hi to the AI
    for update in graph.stream(
        {"messages": [{"role": "user", "content": "hi"}]}, config, stream_mode="updates"
    ):
        print(update)
    

We can access the `in_memory_store` and the `user_id` in _any node_ by passing
`store: BaseStore` and `config: RunnableConfig` as node arguments. Here's how
we might use semantic search in a node to find relevant memories:

    
    
    def update_memory(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    
        # Get the user id from the config
        user_id = config["configurable"]["user_id"]
    
        # Namespace the memory
        namespace = (user_id, "memories")
    
        # ... Analyze conversation and create a new memory
    
        # Create a new memory ID
        memory_id = str(uuid.uuid4())
    
        # We create a new memory
        store.put(namespace, memory_id, {"memory": memory})
    

As we showed above, we can also access the store in any node and use the
`store.search` method to get memories. Recall the the memories are returned as
a list of objects that can be converted to a dictionary.

    
    
    memories[-1].dict()
    {'value': {'food_preference': 'I like pizza'},
     'key': '07e0caf4-1631-47b7-b15f-65515d4c1843',
     'namespace': ['1', 'memories'],
     'created_at': '2024-10-02T17:22:31.590602+00:00',
     'updated_at': '2024-10-02T17:22:31.590605+00:00'}
    

We can access the memories and use them in our model call.

    
    
    def call_model(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
        # Get the user id from the config
        user_id = config["configurable"]["user_id"]
    
        # Search based on the most recent message
        memories = store.search(
            namespace,
            query=state["messages"][-1].content,
            limit=3
        )
        info = "\n".join([d.value["memory"] for d in memories])
    
        # ... Use memories in the model call
    

If we create a new thread, we can still access the same memories so long as
the `user_id` is the same.

    
    
    # Invoke the graph
    config = {"configurable": {"thread_id": "2", "user_id": "1"}}
    
    # Let's say hi again
    for update in graph.stream(
        {"messages": [{"role": "user", "content": "hi, tell me about my memories"}]}, config, stream_mode="updates"
    ):
        print(update)
    

When we use the LangGraph Platform, either locally (e.g., in LangGraph Studio)
or with LangGraph Cloud, the base store is available to use by default and
does not need to be specified during graph compilation. To enable semantic
search, however, you **do** need to configure the indexing settings in your
`langgraph.json` file. For example:

    
    
    {
        ...
        "store": {
            "index": {
                "embed": "openai:text-embeddings-3-small",
                "dims": 1536,
                "fields": ["$"]
            }
        }
    }
    

See the deployment guide for more details and configuration options.

## Checkpointer libraries¶

Under the hood, checkpointing is powered by checkpointer objects that conform
to BaseCheckpointSaver interface. LangGraph provides several checkpointer
implementations, all implemented via standalone, installable libraries:

  * `langgraph-checkpoint`: The base interface for checkpointer savers (BaseCheckpointSaver) and serialization/deserialization interface (SerializerProtocol). Includes in-memory checkpointer implementation (MemorySaver) for experimentation. LangGraph comes with `langgraph-checkpoint` included.
  * `langgraph-checkpoint-sqlite`: An implementation of LangGraph checkpointer that uses SQLite database (SqliteSaver / AsyncSqliteSaver). Ideal for experimentation and local workflows. Needs to be installed separately.
  * `langgraph-checkpoint-postgres`: An advanced checkpointer that uses Postgres database (PostgresSaver / AsyncPostgresSaver), used in LangGraph Cloud. Ideal for using in production. Needs to be installed separately.

### Checkpointer interface¶

Each checkpointer conforms to BaseCheckpointSaver interface and implements the
following methods:

  * `.put` \- Store a checkpoint with its configuration and metadata. 
  * `.put_writes` \- Store intermediate writes linked to a checkpoint (i.e. pending writes). 
  * `.get_tuple` \- Fetch a checkpoint tuple using for a given configuration (`thread_id` and `checkpoint_id`). This is used to populate `StateSnapshot` in `graph.get_state()`. 
  * `.list` \- List checkpoints that match a given configuration and filter criteria. This is used to populate state history in `graph.get_state_history()`

If the checkpointer is used with asynchronous graph execution (i.e. executing
the graph via `.ainvoke`, `.astream`, `.abatch`), asynchronous versions of the
above methods will be used (`.aput`, `.aput_writes`, `.aget_tuple`, `.alist`).

Note

For running your graph asynchronously, you can use `MemorySaver`, or async
versions of Sqlite/Postgres checkpointers -- `AsyncSqliteSaver` /
`AsyncPostgresSaver` checkpointers.

### Serializer¶

When checkpointers save the graph state, they need to serialize the channel
values in the state. This is done using serializer objects.
`langgraph_checkpoint` defines protocol for implementing serializers provides
a default implementation (JsonPlusSerializer) that handles a wide variety of
types, including LangChain and LangGraph primitives, datetimes, enums and
more.

## Capabilities¶

### Human-in-the-loop¶

First, checkpointers facilitate human-in-the-loop workflows workflows by
allowing humans to inspect, interrupt, and approve graph steps. Checkpointers
are needed for these workflows as the human has to be able to view the state
of a graph at any point in time, and the graph has to be to resume execution
after the human has made any updates to the state. See these how-to guides for
concrete examples.

### Memory¶

Second, checkpointers allow for "memory" between interactions. In the case of
repeated human interactions (like conversations) any follow up messages can be
sent to that thread, which will retain its memory of previous ones. See this
how-to guide for an end-to-end example on how to add and manage conversation
memory using checkpointers.

### Time Travel¶

Third, checkpointers allow for "time travel", allowing users to replay prior
graph executions to review and / or debug specific graph steps. In addition,
checkpointers make it possible to fork the graph state at arbitrary
checkpoints to explore alternative trajectories.

### Fault-tolerance¶

Lastly, checkpointing also provides fault-tolerance and error recovery: if one
or more nodes fail at a given superstep, you can restart your graph from the
last successful step. Additionally, when a graph node fails mid-execution at a
given superstep, LangGraph stores pending checkpoint writes from any other
nodes that completed successfully at that superstep, so that whenever we
resume graph execution from that superstep we don't re-run the successful
nodes.

#### Pending writes¶

Additionally, when a graph node fails mid-execution at a given superstep,
LangGraph stores pending checkpoint writes from any other nodes that completed
successfully at that superstep, so that whenever we resume graph execution
from that superstep we don't re-run the successful nodes.

## Comments

Back to top

Previous

Human-in-the-loop

Next

Memory

Made with  Material for MkDocs Insiders



Skip to content

To learn more about LangGraph, check out our first LangChain Academy course,
_Introduction to LangGraph_ , available for free here.

![logo](./Streaming_files/wordmark_dark.svg)
![logo](./Streaming_files/wordmark_light.svg)

Streaming

Type to start searching

GitHub

  * 0.2.62
  * 7.9k
  * 1.3k

  * Home 
  * Tutorials 
  * How-to Guides 
  * Conceptual Guides 
  * Reference 

![logo](./Streaming_files/wordmark_dark.svg)
![logo](./Streaming_files/wordmark_light.svg)

GitHub

  * 0.2.62
  * 7.9k
  * 1.3k

  * Home 
  * Tutorials 
  * How-to Guides 
  * Conceptual Guides 

Conceptual Guides

    * LangGraph  LangGraph 
      * LangGraph 
      * Why LangGraph? 
      * LangGraph Glossary 
      * Agent architectures 
      * Multi-agent Systems 
      * Human-in-the-loop 
      * Persistence 
      * Memory 
      * Streaming  Streaming  Table of contents 
        * Streaming graph outputs (.stream and .astream) 
        * Streaming LLM tokens and events (.astream_events) 
        * LangGraph Platform 
      * FAQ 
    * LangGraph Platform  LangGraph Platform 
      * LangGraph Platform 
      * High Level 
      * Components 
      * LangGraph Server 
      * Deployment Options 
  * Reference 

Table of contents

  * Streaming graph outputs (.stream and .astream) 
  * Streaming LLM tokens and events (.astream_events) 
  * LangGraph Platform 

  1. Home 
  2. Conceptual Guides 
  3. LangGraph 

# Streaming¶

LangGraph is built with first class support for streaming. There are several
different ways to stream back outputs from a graph run

## Streaming graph outputs (`.stream` and `.astream`)¶

`.stream` and `.astream` are sync and async methods for streaming back outputs
from a graph run. There are several different modes you can specify when
calling these methods (e.g. `graph.stream(..., mode="...")):

  * `"values"`: This streams the full value of the state after each step of the graph.
  * `"updates"`: This streams the updates to the state after each step of the graph. If multiple updates are made in the same step (e.g. multiple nodes are run) then those updates are streamed separately.
  * `"custom"`: This streams custom data from inside your graph nodes.
  * `"messages"`: This streams LLM tokens and metadata for the graph node where LLM is invoked.
  * `"debug"`: This streams as much information as possible throughout the execution of the graph.

You can also specify multiple streaming modes at the same time by passing them
as a list. When you do this, the streamed outputs will be tuples
`(stream_mode, data)`. For example:

    
    
    graph.stream(..., stream_mode=["updates", "messages"])
    
    
    
    ...
    ('messages', (AIMessageChunk(content='Hi'), {'langgraph_step': 3, 'langgraph_node': 'agent', ...}))
    ...
    ('updates', {'agent': {'messages': [AIMessage(content="Hi, how can I help you?")]}})
    

The below visualization shows the difference between the `values` and
`updates` modes:

![values vs updates](./Streaming_files/values_vs_updates.png)

## Streaming LLM tokens and events (`.astream_events`)¶

In addition, you can use the `astream_events` method to stream back events
that happen _inside_ nodes. This is useful for streaming tokens of LLM calls.

This is a standard method on all LangChain objects. This means that as the
graph is executed, certain events are emitted along the way and can be seen if
you run the graph using `.astream_events`.

All events have (among other things) `event`, `name`, and `data` fields. What
do these mean?

  * `event`: This is the type of event that is being emitted. You can find a detailed table of all callback events and triggers here.
  * `name`: This is the name of event.
  * `data`: This is the data associated with the event.

What types of things cause events to be emitted?

  * each node (runnable) emits `on_chain_start` when it starts execution, `on_chain_stream` during the node execution and `on_chain_end` when the node finishes. Node events will have the node name in the event's `name` field
  * the graph will emit `on_chain_start` in the beginning of the graph execution, `on_chain_stream` after each node execution and `on_chain_end` when the graph finishes. Graph events will have the `LangGraph` in the event's `name` field
  * Any writes to state channels (i.e. anytime you update the value of one of your state keys) will emit `on_chain_start` and `on_chain_end` events

Additionally, any events that are created inside your nodes (LLM events, tool
events, manually emitted events, etc.) will also be visible in the output of
`.astream_events`.

To make this more concrete and to see what this looks like, let's see what
events are returned when we run a simple graph:

    
    
    from langchain_openai import ChatOpenAI
    from langgraph.graph import StateGraph, MessagesState, START, END
    
    model = ChatOpenAI(model="gpt-4o-mini")
    
    
    def call_model(state: MessagesState):
        response = model.invoke(state['messages'])
        return {"messages": response}
    
    workflow = StateGraph(MessagesState)
    workflow.add_node(call_model)
    workflow.add_edge(START, "call_model")
    workflow.add_edge("call_model", END)
    app = workflow.compile()
    
    inputs = [{"role": "user", "content": "hi!"}]
    async for event in app.astream_events({"messages": inputs}, version="v1"):
        kind = event["event"]
        print(f"{kind}: {event['name']}")
    

API Reference: ChatOpenAI | StateGraph | START | END 
    
    
    on_chain_start: LangGraph
    on_chain_start: __start__
    on_chain_end: __start__
    on_chain_start: call_model
    on_chat_model_start: ChatOpenAI
    on_chat_model_stream: ChatOpenAI
    on_chat_model_stream: ChatOpenAI
    on_chat_model_stream: ChatOpenAI
    on_chat_model_stream: ChatOpenAI
    on_chat_model_stream: ChatOpenAI
    on_chat_model_stream: ChatOpenAI
    on_chat_model_stream: ChatOpenAI
    on_chat_model_stream: ChatOpenAI
    on_chat_model_stream: ChatOpenAI
    on_chat_model_stream: ChatOpenAI
    on_chat_model_stream: ChatOpenAI
    on_chat_model_end: ChatOpenAI
    on_chain_start: ChannelWrite<call_model,messages>
    on_chain_end: ChannelWrite<call_model,messages>
    on_chain_stream: call_model
    on_chain_end: call_model
    on_chain_stream: LangGraph
    on_chain_end: LangGraph
    

We start with the overall graph start (`on_chain_start: LangGraph`). We then
write to the `__start__` node (this is special node to handle input). We then
start the `call_model` node (`on_chain_start: call_model`). We then start the
chat model invocation (`on_chat_model_start: ChatOpenAI`), stream back token
by token (`on_chat_model_stream: ChatOpenAI`) and then finish the chat model
(`on_chat_model_end: ChatOpenAI`). From there, we write the results back to
the channel (`ChannelWrite<call_model,messages>`) and then finish the
`call_model` node and then the graph as a whole.

This should hopefully give you a good sense of what events are emitted in a
simple graph. But what data do these events contain? Each type of event
contains data in a different format. Let's look at what `on_chat_model_stream`
events look like. This is an important type of event since it is needed for
streaming tokens from an LLM response.

These events look like:

    
    
    {'event': 'on_chat_model_stream',
     'name': 'ChatOpenAI',
     'run_id': '3fdbf494-acce-402e-9b50-4eab46403859',
     'tags': ['seq:step:1'],
     'metadata': {'langgraph_step': 1,
      'langgraph_node': 'call_model',
      'langgraph_triggers': ['start:call_model'],
      'langgraph_task_idx': 0,
      'checkpoint_id': '1ef657a0-0f9d-61b8-bffe-0c39e4f9ad6c',
      'checkpoint_ns': 'call_model',
      'ls_provider': 'openai',
      'ls_model_name': 'gpt-4o-mini',
      'ls_model_type': 'chat',
      'ls_temperature': 0.7},
     'data': {'chunk': AIMessageChunk(content='Hello', id='run-3fdbf494-acce-402e-9b50-4eab46403859')},
     'parent_ids': []}
    

We can see that we have the event type and name (which we knew from before).

We also have a bunch of stuff in metadata. Noticeably, `'langgraph_node':
'call_model',` is some really helpful information which tells us which node
this model was invoked inside of.

Finally, `data` is a really important field. This contains the actual data for
this event! Which in this case is an AIMessageChunk. This contains the
`content` for the message, as well as an `id`. This is the ID of the overall
AIMessage (not just this chunk) and is super helpful - it helps us track which
chunks are part of the same message (so we can show them together in the UI).

This information contains all that is needed for creating a UI for streaming
LLM tokens. You can see a guide for that here.

ASYNC IN PYTHON<=3.10

You may fail to see events being emitted from inside a node when using
`.astream_events` in Python <= 3.10. If you're using a Langchain
RunnableLambda, a RunnableGenerator, or Tool asynchronously inside your node,
you will have to propagate callbacks to these objects manually. This is
because LangChain cannot automatically propagate callbacks to child objects in
this case. Please see examples here and here.

## LangGraph Platform¶

Streaming is critical for making LLM applications feel responsive to end
users. When creating a streaming run, the streaming mode determines what data
is streamed back to the API client. LangGraph Platform supports five streaming
modes:

  * `values`: Stream the full state of the graph after each super-step is executed. See the how-to guide for streaming values.
  * `messages-tuple`: Stream LLM tokens for any messages generated inside a node. This mode is primarily meant for powering chat applications. See the how-to guide for streaming messages.
  * `updates`: Streams updates to the state of the graph after each node is executed. See the how-to guide for streaming updates.
  * `events`: Stream all events (including the state of the graph) that occur during graph execution. See the how-to guide for streaming events. This can be used to do token-by-token streaming for LLMs.
  * `debug`: Stream debug events throughout graph execution. See the how-to guide for streaming debug events.

You can also specify multiple streaming modes at the same time. See the how-to
guide for configuring multiple streaming modes at the same time.

See the API reference for how to create streaming runs.

Streaming modes `values`, `updates`, `messages-tuple` and `debug` are very
similar to modes available in the LangGraph library - for a deeper conceptual
explanation of those, you can see the previous section.

Streaming mode `events` is the same as using `.astream_events` in the
LangGraph library - for a deeper conceptual explanation of this, you can see
the previous section.

All events emitted have two attributes:

  * `event`: This is the name of the event
  * `data`: This is data associated with the event

## Comments

Back to top

Previous

Memory

Next

FAQ

Made with  Material for MkDocs Insiders



Skip to content

To learn more about LangGraph, check out our first LangChain Academy course,
_Introduction to LangGraph_ , available for free here.

![logo](./Time Travel ⏱️_files/wordmark_dark.svg) ![logo](./Time Travel
⏱️_files/wordmark_light.svg)

Time Travel ⏱️

Type to start searching

GitHub

  * 0.2.62
  * 7.9k
  * 1.3k

  * Home 
  * Tutorials 
  * How-to Guides 
  * Conceptual Guides 
  * Reference 

![logo](./Time Travel ⏱️_files/wordmark_dark.svg) ![logo](./Time Travel
⏱️_files/wordmark_light.svg)

GitHub

  * 0.2.62
  * 7.9k
  * 1.3k

  * Home 
  * Tutorials 
  * How-to Guides 
  * Conceptual Guides 
  * Reference 

Table of contents

  * Replaying 
  * Forking 
  * Additional Resources 📚 

# Time Travel ⏱️¶

Prerequisites

This guide assumes that you are familiar with LangGraph's checkpoints and
states. If not, please review the persistence concept first.

When working with non-deterministic systems that make model-based decisions
(e.g., agents powered by LLMs), it can be useful to examine their decision-
making process in detail:

  1. 🤔 **Understand Reasoning** : Analyze the steps that led to a successful result.
  2. 🐞 **Debug Mistakes** : Identify where and why errors occurred.
  3. 🔍 **Explore Alternatives** : Test different paths to uncover better solutions.

We call these debugging techniques **Time Travel** , composed of two key
actions: **Replaying** 🔁 and **Forking** 🔀 .

## Replaying¶

![](./Time Travel ⏱️_files/replay.png)

Replaying allows us to revisit and reproduce an agent's past actions. This can
be done either from the current state (or checkpoint) of the graph or from a
specific checkpoint.

To replay from the current state, simply pass `None` as the input along with a
`thread`:

    
    
    thread = {"configurable": {"thread_id": "1"}}
    for event in graph.stream(None, thread, stream_mode="values"):
        print(event)
    

To replay actions from a specific checkpoint, start by retrieving all
checkpoints for the thread:

    
    
    all_checkpoints = []
    for state in graph.get_state_history(thread):
        all_checkpoints.append(state)
    

Each checkpoint has a unique ID. After identifying the desired checkpoint, for
instance, `xyz`, include its ID in the configuration:

    
    
    config = {'configurable': {'thread_id': '1', 'checkpoint_id': 'xyz'}}
    for event in graph.stream(None, config, stream_mode="values"):
        print(event)
    

The graph efficiently replays previously executed nodes instead of re-
executing them, leveraging its awareness of prior checkpoint executions.

## Forking¶

![](./Time Travel ⏱️_files/forking.png)

Forking allows you to revisit an agent's past actions and explore alternative
paths within the graph.

To edit a specific checkpoint, such as `xyz`, provide its `checkpoint_id` when
updating the graph's state:

    
    
    config = {"configurable": {"thread_id": "1", "checkpoint_id": "xyz"}}
    graph.update_state(config, {"state": "updated state"})
    

This creates a new forked checkpoint, xyz-fork, from which you can continue
running the graph:

    
    
    config = {'configurable': {'thread_id': '1', 'checkpoint_id': 'xyz-fork'}}
    for event in graph.stream(None, config, stream_mode="values"):
        print(event)
    

## Additional Resources 📚¶

  * **Conceptual Guide: Persistence** : Read the persistence guide for more context on replaying.
  * **How to View and Update Past Graph State** : Step-by-step instructions for working with graph state that demonstrate the **replay** and **fork** actions.

## Comments

Back to top

Made with  Material for MkDocs Insiders



Skip to content

To learn more about LangGraph, check out our first LangChain Academy course,
_Introduction to LangGraph_ , available for free here.

![logo](./Why LangGraph__files/wordmark_dark.svg) ![logo](./Why
LangGraph__files/wordmark_light.svg)

Why LangGraph?

Type to start searching

GitHub

  * 0.2.62
  * 7.9k
  * 1.3k

Go to repository

  * Home 
  * Tutorials 
  * How-to Guides 
  * Conceptual Guides 
  * Reference 

![logo](./Why LangGraph__files/wordmark_dark.svg) ![logo](./Why
LangGraph__files/wordmark_light.svg)

GitHub

  * 0.2.62
  * 7.9k
  * 1.3k

  * Home 
  * Tutorials 
  * How-to Guides 
  * Conceptual Guides 

Conceptual Guides

    * LangGraph  LangGraph 
      * LangGraph 
      * Why LangGraph?  Why LangGraph?  Table of contents 
        * Challenges 
        * Core Principles 
        * Debugging 
        * Deployment 
      * LangGraph Glossary 
      * Agent architectures 
      * Multi-agent Systems 
      * Human-in-the-loop 
      * Persistence 
      * Memory 
      * Streaming 
      * FAQ 
    * LangGraph Platform  LangGraph Platform 
      * LangGraph Platform 
      * High Level 
      * Components 
      * LangGraph Server 
      * Deployment Options 
  * Reference 

Table of contents

  * Challenges 
  * Core Principles 
  * Debugging 
  * Deployment 

  1. Home 
  2. Conceptual Guides 
  3. LangGraph 

# Why LangGraph?¶

LLMs are extremely powerful, particularly when connected to other systems such
as a retriever or APIs. This is why many LLM applications use a control flow
of steps before and / or after LLM calls. As an example RAG performs retrieval
of relevant documents to a question, and passes those documents to an LLM in
order to ground the response. Often a control flow of steps before and / or
after an LLM is called a "chain." Chains are a popular paradigm for
programming with LLMs and offer a high degree of reliability; the same set of
steps runs with each chain invocation.

However, we often want LLM systems that can pick their own control flow! This
is one definition of an agent: an agent is a system that uses an LLM to decide
the control flow of an application. Unlike a chain, an agent gives an LLM some
degree of control over the sequence of steps in the application. Examples of
using an LLM to decide the control of an application:

  * Using an LLM to route between two potential paths
  * Using an LLM to decide which of many tools to call
  * Using an LLM to decide whether the generated answer is sufficient or more work is need

There are many different types of agent architectures to consider, which give
an LLM varying levels of control. On one extreme, a router allows an LLM to
select a single step from a specified set of options and, on the other
extreme, a fully autonomous long-running agent may have complete freedom to
select any sequence of steps that it wants for a given problem.

![Agent Types](./Why LangGraph__files/agent_types.png)

Several concepts are utilized in many agent architectures:

  * Tool calling: this is often how LLMs make decisions
  * Action taking: often times, the LLMs' outputs are used as the input to an action
  * Memory: reliable systems need to have knowledge of things that occurred
  * Planning: planning steps (either explicit or implicit) are useful for ensuring that the LLM, when making decisions, makes them in the highest fidelity way.

## Challenges¶

In practice, there is often a trade-off between control and reliability. As we
give LLMs more control, the application often become less reliable. This can
be due to factors such as LLM non-determinism and / or errors in selecting
tools (or steps) that the agent uses (takes).

![Agent Challenge](./Why LangGraph__files/challenge.png)

## Core Principles¶

The motivation of LangGraph is to help bend the curve, preserving higher
reliability as we give the agent more control over the application. We'll
outline a few specific pillars of LangGraph that make it well suited for
building reliable agents.

![Langgraph](./Why LangGraph__files/langgraph.png)

**Controllability**

LangGraph gives the developer a high degree of control by expressing the flow
of the application as a set of nodes and edges. All nodes can access and
modify a common state (memory). The control flow of the application can set
using edges that connect nodes, either deterministically or via conditional
logic.

**Persistence**

LangGraph gives the developer many options for persisting graph state using
short-term or long-term (e.g., via a database) memory.

**Human-in-the-Loop**

The persistence layer enables several different human-in-the-loop interaction
patterns with agents; for example, it's possible to pause an agent, review its
state, edit it state, and approve a follow-up step.

**Streaming**

LangGraph comes with first class support for streaming, which can expose state
to the user (or developer) over the course of agent execution. LangGraph
supports streaming of both events (like a tool call being taken) as well as of
tokens that an LLM may emit.

## Debugging¶

Once you've built a graph, you often want to test and debug it. LangGraph
Studio is a specialized IDE for visualization and debugging of LangGraph
applications.

![Langgraph Studio](./Why LangGraph__files/lg_studio.png)

## Deployment¶

Once you have confidence in your LangGraph application, many developers want
an easy path to deployment. LangGraph Platform offers a range of options for
deploying LangGraph graphs.

## Comments

Back to top

Previous

Concepts

Next

LangGraph Glossary

Made with  Material for MkDocs Insiders



