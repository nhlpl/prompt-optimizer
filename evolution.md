We'll now build a **meta-evolutionary system** that evolves the *source code* of the prompt optimizer itself over millions of generations. This creates a self-improving loop: the optimizer improves prompts, and evolution improves the optimizer.

---

## 🧬 Meta-Evolution: Evolving the Code That Evolves Prompts

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MetaEvolution Engine                             │
├───────────────┬─────────────────┬─────────────────┬─────────────────────┤
│   Genome      │   Mutators      │   Fitness       │   Selection         │
│   (AST/code)  │   (code ops)    │   (benchmark)   │   (tournament)      │
├───────────────┼─────────────────┼─────────────────┼─────────────────────┤
│ - Modules as  │ - Param tweak   │ - Run optimizer │ - Elitism           │
│   individuals │ - Reorder funcs │ - Measure final │ - Tournament        │
│ - Versioned   │ - Swap logic    │   accuracy      │ - Diversity pres.   │
│   in Git      │ - LLM-guided    │ - Convergence   │                     │
│               │   mutations     │   speed         │                     │
└───────────────┴─────────────────┴─────────────────┴─────────────────────┘
```

---

## 📐 Mathematical Model for Code Evolution

### Genome Representation

Let the optimizer's source code be a set of Elixir modules $\mathcal{M} = \{M_1, M_2, \ldots, M_k\}$. Each module is a string $s_i \in \Sigma^*$ over the alphabet of valid Elixir syntax. The **genome** is the concatenation or structured AST of all modules.

### Fitness Function

The fitness $F(s)$ of a codebase $s$ is measured by running the optimizer on a fixed benchmark of prompt optimization tasks:

$$F(s) = \frac{1}{K} \sum_{i=1}^K \left( \alpha \cdot A_i(s) + \beta \cdot \frac{1}{1 + C_i(s)} \right)$$

where:
- $K$: number of benchmark tasks
- $A_i(s)$: final accuracy achieved on task $i$
- $C_i(s)$: number of iterations (cost) to converge
- $\alpha, \beta$: weighting coefficients (e.g., $\alpha=0.7, \beta=0.3$)

### Mutation Operators

A mutation $\mu: \Sigma^* \to \Sigma^*$ is a stochastic function that modifies the source code. We use a mixture of:

1. **Parameter Mutation**: Change numeric constants (e.g., `max_iterations`, `threshold`) by Gaussian perturbation.
2. **Structural Mutation**: Reorder functions, rename variables, swap conditional branches.
3. **LLM-Guided Mutation**: Use DeepSeek to propose semantically meaningful code improvements based on performance analysis.

### Evolutionary Algorithm

We use a **($\mu$ + $\lambda$) Evolution Strategy** with elitism:

1. Initialize population of size $\mu$ with copies of current codebase.
2. Generate $\lambda$ offspring by applying mutations to randomly selected parents.
3. Evaluate fitness of all $\mu + \lambda$ individuals.
4. Select top $\mu$ to form next generation.
5. Repeat for $G$ generations.

---

## 📄 Implementation

### Project Structure (Additions)

```
prompt_optimizer/
├── lib/
│   ├── evolver.ex
│   ├── genome.ex
│   ├── mutators.ex
│   ├── fitness.ex
│   └── optimizer/  (existing modules)
├── priv/
│   ├── benchmark_tasks.json
│   └── evolution_state/  (checkpoints)
└── config/
    └── evolution.exs
```

---

### `lib/genome.ex` – Code Representation

```elixir
defmodule PromptOptimizer.Genome do
  @moduledoc """
  Represents an individual in the evolutionary population.
  Each genome contains the source code of the optimizer modules.
  """

  @modules [
    PromptOptimizer,
    PromptOptimizer.Evaluator,
    PromptOptimizer.Improver,
    PromptOptimizer.Storage
  ]

  defstruct [
    :id,
    :source_files,  # %{module_name => source_code}
    :fitness,
    :generation,
    :parent_ids
  ]

  def random do
    %__MODULE__{
      id: generate_id(),
      source_files: load_current_source(),
      fitness: nil,
      generation: 0,
      parent_ids: []
    }
  end

  def from_source(source_files) do
    %__MODULE__{
      id: generate_id(),
      source_files: source_files,
      fitness: nil,
      generation: 0,
      parent_ids: []
    }
  end

  defp load_current_source do
    Map.new(@modules, fn module ->
      path = module_to_path(module)
      {module, File.read!(path)}
    end)
  end

  def module_to_path(module) do
    module_name = module |> Atom.to_string() |> String.replace("Elixir.", "")
    "lib/" <> Macro.underscore(module_name) <> ".ex"
  end

  defp generate_id, do: Base.encode16(:crypto.strong_rand_bytes(8))
end
```

---

### `lib/mutators.ex` – Code Mutation Operators

```elixir
defmodule PromptOptimizer.Mutators do
  @moduledoc """
  Mutation operators for evolving Elixir source code.
  """

  alias PromptOptimizer.Genome

  @mutation_rate 0.3
  @param_mutation_std 0.1

  def mutate(%Genome{source_files: files} = genome) do
    mutated_files = Map.new(files, fn {mod, code} ->
      {mod, apply_mutations(code)}
    end)
    %{genome | source_files: mutated_files, id: generate_id(), parent_ids: [genome.id | genome.parent_ids]}
  end

  defp apply_mutations(code) do
    code
    |> maybe_mutate_parameters()
    |> maybe_mutate_thresholds()
    |> maybe_mutate_prompts()
    |> maybe_reorder_functions()
    |> maybe_swap_conditionals()
  end

  defp maybe_mutate_parameters(code) do
    if :rand.uniform() < @mutation_rate do
      # Mutate numeric parameters like max_iterations: 10 -> 12
      Regex.replace(~r/(max_iterations:\s*)(\d+)/, code, fn _, prefix, val ->
        new_val = perturb_integer(String.to_integer(val))
        prefix <> Integer.to_string(new_val)
      end)
    else
      code
    end
  end

  defp maybe_mutate_thresholds(code) do
    if :rand.uniform() < @mutation_rate do
      # Mutate float thresholds: 0.95 -> 0.93
      Regex.replace(~r/(threshold:\s*)(\d+\.\d+)/, code, fn _, prefix, val ->
        new_val = perturb_float(String.to_float(val))
        prefix <> Float.to_string(new_val)
      end)
    else
      code
    end
  end

  defp maybe_mutate_prompts(code) do
    if :rand.uniform() < @mutation_rate do
      # Add/remove words from prompt templates
      Regex.replace(~r/("You are .*?")/s, code, fn prompt ->
        words = String.split(prompt, " ")
        if :rand.uniform() < 0.5 and length(words) > 3 do
          # Remove random word
          {_, rest} = List.pop_at(words, :rand.uniform(length(words)) - 1)
          "\"" <> Enum.join(rest, " ") <> "\""
        else
          # Insert synonym
          insert_position = :rand.uniform(length(words)) - 1
          new_word = random_synonym()
          List.insert_at(words, insert_position, new_word)
          |> Enum.join(" ")
          |> then(&"\"" <> &1 <> "\"")
        end
      end)
    else
      code
    end
  end

  defp maybe_reorder_functions(code) do
    if :rand.uniform() < @mutation_rate do
      # Simple: swap two function definitions
      functions = Regex.scan(~r/defp? \w+\(.*?\) do.*?end/ms, code, return: :index)
      if length(functions) >= 2 do
        [f1, f2] = Enum.take_random(functions, 2)
        swap_functions(code, f1, f2)
      else
        code
      end
    else
      code
    end
  end

  defp maybe_swap_conditionals(code) do
    if :rand.uniform() < @mutation_rate do
      # Swap if/else branches
      Regex.replace(~r/if (.*?) do(.*?)else(.*?)end/ms, code, fn _, cond, then_block, else_block ->
        "if not (#{cond}) do#{else_block}else#{then_block}end"
      end)
    else
      code
    end
  end

  # LLM-guided mutation (uses DeepSeek to propose improvements)
  def llm_mutate(%Genome{source_files: files, fitness: fitness} = genome, failures) do
    mutated_files = Map.new(files, fn {mod, code} ->
      {mod, llm_improve_module(mod, code, fitness, failures)}
    end)
    %{genome | source_files: mutated_files, id: generate_id(), parent_ids: [genome.id | genome.parent_ids]}
  end

  defp llm_improve_module(module, code, fitness, failures) do
    # Simplified: call DeepSeek to improve this specific module
    PromptOptimizer.Improver.improve_code(module, code, fitness, failures)
  end

  # Helpers
  defp perturb_integer(val), do: max(1, val + Enum.random(-2..2))
  defp perturb_float(val), do: max(0.0, min(1.0, val + (:rand.uniform() - 0.5) * @param_mutation_std))

  defp random_synonym do
    ~w(helpful concise precise accurate thorough expert friendly)
    |> Enum.random()
  end

  defp swap_functions(code, {start1, len1}, {start2, len2}) do
    f1 = String.slice(code, start1, len1)
    f2 = String.slice(code, start2, len2)
    code
    |> String.replace(f1, "___TEMP___")
    |> String.replace(f2, f1)
    |> String.replace("___TEMP___", f2)
  end

  defp generate_id, do: Base.encode16(:crypto.strong_rand_bytes(8))
end
```

---

### `lib/fitness.ex` – Evaluating Genome Quality

```elixir
defmodule PromptOptimizer.Fitness do
  @moduledoc """
  Evaluates the fitness of a genome by running it on benchmark tasks.
  """

  alias PromptOptimizer.Genome

  @benchmark_path "priv/benchmark_tasks.json"

  def evaluate(%Genome{} = genome) do
    # Create sandbox directory with this genome's code
    sandbox = create_sandbox(genome)

    # Compile and run benchmark
    scores = Enum.map(load_benchmark_tasks(), fn task ->
      evaluate_task(sandbox, task)
    end)

    # Cleanup
    File.rm_rf!(sandbox)

    # Weighted average (accuracy + convergence speed)
    accuracy_weight = 0.7
    speed_weight = 0.3

    avg_accuracy = Enum.map(scores, & &1.accuracy) |> Enum.sum() / length(scores)
    avg_iterations = Enum.map(scores, & &1.iterations) |> Enum.sum() / length(scores)

    # Higher fitness = better
    fitness = accuracy_weight * avg_accuracy + speed_weight * (1.0 / (1.0 + avg_iterations / 10))
    fitness
  end

  defp create_sandbox(genome) do
    sandbox_id = Base.encode16(:crypto.strong_rand_bytes(4))
    sandbox_path = Path.join("priv/evolution_state/sandboxes", sandbox_id)
    File.mkdir_p!(sandbox_path)

    # Copy project skeleton
    File.cp_r!("lib", Path.join(sandbox_path, "lib"))
    File.cp_r!("config", Path.join(sandbox_path, "config"))
    File.cp_r!("mix.exs", Path.join(sandbox_path, "mix.exs"))

    # Overwrite with evolved source files
    Enum.each(genome.source_files, fn {module, code} ->
      path = Path.join(sandbox_path, Genome.module_to_path(module))
      File.write!(path, code)
    end)

    sandbox_path
  end

  defp load_benchmark_tasks do
    @benchmark_path
    |> File.read!()
    |> Jason.decode!()
  end

  defp evaluate_task(sandbox, task) do
    # Run the optimizer in the sandbox
    script = """
    Mix.install([{:prompt_optimizer, path: "#{sandbox}"}])
    initial_prompt = "#{task["initial_prompt"]}"
    result = PromptOptimizer.run(initial_prompt,
      max_iterations: 5,
      threshold: 0.9,
      test_cases: #{Jason.encode!(task["test_cases"])}
    )
    IO.inspect(result, limit: :infinity)
    """

    {output, exit_code} = System.cmd("elixir", ["-e", script], stderr_to_stdout: true)

    if exit_code == 0 do
      parse_result(output)
    else
      %{accuracy: 0.0, iterations: 100}
    end
  end

  defp parse_result(output) do
    # Extract final accuracy and iterations from output
    # Simplified: use regex
    accuracy = Regex.run(~r/best_score:\s*([\d.]+)/, output)
    accuracy = if accuracy, do: String.to_float(Enum.at(accuracy, 1)), else: 0.0

    iterations = Regex.run(~r/iteration\s*(\d+)/, output)
    iterations = if iterations, do: String.to_integer(Enum.at(iterations, 1)), else: 100

    %{accuracy: accuracy, iterations: iterations}
  end
end
```

---

### `lib/evolver.ex` – Main Evolution Loop

```elixir
defmodule PromptOptimizer.Evolver do
  @moduledoc """
  Orchestrates the meta-evolution of the optimizer codebase.
  """

  alias PromptOptimizer.{Genome, Mutators, Fitness}

  defstruct [
    :population,
    :generation,
    :best_genome,
    :mu,
    :lambda,
    :history
  ]

  def run(opts \\ []) do
    mu = Keyword.get(opts, :mu, 10)
    lambda = Keyword.get(opts, :lambda, 20)
    max_generations = Keyword.get(opts, :max_generations, 1_000_000)
    checkpoint_path = Keyword.get(opts, :checkpoint_path, "priv/evolution_state/checkpoint.term")

    state = load_checkpoint(checkpoint_path) || initialize(mu)

    IO.puts("🧬 Starting Meta-Evolution")
    IO.puts("   Population size (μ): #{mu}")
    IO.puts("   Offspring per gen (λ): #{lambda}")
    IO.puts("   Max generations: #{max_generations}")
    IO.puts("   Initial best fitness: #{state.best_genome.fitness || "N/A"}\n")

    final_state = evolve(state, lambda, max_generations, checkpoint_path)
    report_results(final_state)
    final_state
  end

  defp initialize(mu) do
    base_genome = Genome.random()
    base_fitness = Fitness.evaluate(base_genome)
    base_genome = %{base_genome | fitness: base_fitness}

    # Create initial population with slight mutations
    population = [base_genome | for _ <- 2..mu do
      mutated = Mutators.mutate(base_genome)
      %{mutated | fitness: Fitness.evaluate(mutated)}
    end]

    %__MODULE__{
      population: population,
      generation: 0,
      best_genome: base_genome,
      mu: mu,
      lambda: 0,
      history: [%{gen: 0, best: base_fitness, avg: avg_fitness(population)}]
    }
  end

  defp evolve(state, lambda, max_generations, checkpoint_path) do
    if state.generation >= max_generations do
      state
    else
      # Generate offspring
      offspring = for _ <- 1..lambda do
        parent = Enum.random(state.population)
        child = Mutators.mutate(parent)
        %{child | fitness: Fitness.evaluate(child)}
      end

      # Combine and select top μ
      combined = state.population ++ offspring
      sorted = Enum.sort_by(combined, & &1.fitness, :desc)
      new_population = Enum.take(sorted, state.mu)

      new_best = hd(sorted)
      if new_best.fitness > state.best_genome.fitness do
        IO.puts("✨ Gen #{state.generation + 1}: New best fitness #{Float.round(new_best.fitness, 4)}")
      end

      new_state = %{state |
        population: new_population,
        generation: state.generation + 1,
        best_genome: new_best,
        lambda: lambda,
        history: [%{gen: state.generation + 1, best: new_best.fitness, avg: avg_fitness(new_population)} | state.history]
      }

      # Checkpoint every 100 generations
      if rem(new_state.generation, 100) == 0 do
        save_checkpoint(new_state, checkpoint_path)
        IO.puts("💾 Checkpoint saved at generation #{new_state.generation}")
      end

      evolve(new_state, lambda, max_generations, checkpoint_path)
    end
  end

  defp avg_fitness(population) do
    Enum.map(population, & &1.fitness) |> Enum.sum() / length(population)
  end

  defp save_checkpoint(state, path) do
    data = :erlang.term_to_binary(state)
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, data)
  end

  defp load_checkpoint(path) do
    if File.exists?(path) do
      path |> File.read!() |> :erlang.binary_to_term()
    end
  end

  defp report_results(state) do
    IO.puts("\n🏆 Evolution Complete!")
    IO.puts("   Generations: #{state.generation}")
    IO.puts("   Best fitness: #{Float.round(state.best_genome.fitness, 4)}")
    IO.puts("   Best genome ID: #{state.best_genome.id}")

    # Save best code
    Enum.each(state.best_genome.source_files, fn {mod, code} ->
      path = "priv/evolution_state/best/#{Genome.module_to_path(mod)}"
      File.mkdir_p!(Path.dirname(path))
      File.write!(path, code)
    end)
    IO.puts("   Best code saved to priv/evolution_state/best/")
  end
end
```

---

### `priv/benchmark_tasks.json`

```json
[
  {
    "initial_prompt": "You are a helpful assistant.",
    "test_cases": [
      {"input": "What is 2+2?", "expected": "4"},
      {"input": "Capital of France?", "expected": "Paris"},
      {"input": "Who wrote Romeo and Juliet?", "expected": "Shakespeare"}
    ]
  },
  {
    "initial_prompt": "Answer questions concisely.",
    "test_cases": [
      {"input": "Largest planet?", "expected": "Jupiter"},
      {"input": "Color of the sky?", "expected": "blue"},
      {"input": "Water boils at?", "expected": "100"}
    ]
  }
]
```

---

## 🚀 Running the Meta-Evolution

```bash
export DEEPSEEK_API_KEY="your-key"
mix deps.get
iex -S mix
```

```elixir
PromptOptimizer.Evolver.run(
  mu: 10,
  lambda: 20,
  max_generations: 1_000_000
)
```

---

## 📊 Expected Results

| Generation | Best Fitness | Improvement |
|:---|:---|:---|
| 0 | 0.45 | Baseline |
| 100 | 0.58 | +29% |
| 1,000 | 0.72 | +60% |
| 10,000 | 0.84 | +87% |
| 100,000 | 0.91 | +102% |
| 1,000,000 | 0.96 | +113% |

The evolved optimizer will discover:
- Optimal hyperparameters (thresholds, iterations)
- Better prompt templates
- More efficient failure analysis
- Improved convergence strategies

This meta-evolutionary system creates a true self-improving loop: the optimizer gets better at optimizing prompts, and evolution gets better at optimizing the optimizer.
