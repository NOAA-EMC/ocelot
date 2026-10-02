# OCELOT Engineering Guidelines

## Objective

Keep OCELOT easy to understand and extend. Prefer cohesive components with clear
ownership over large functions that accumulate configuration branches. New code
must fit the configuration-driven architecture rather than bypass it.

## Before Editing

For any non-trivial change:

1. Locate the class or module that owns the behavior, its configuration class,
   its factory or registry, and a neighboring implementation or test.
2. State the responsibility being changed and the smallest coherent design for
   changing it. Do not begin by appending logic to the first call site found.
3. Decide whether the variation is data or behavior:
   - Put values, thresholds, feature lists, and component selection in typed
     configuration.
   - Put different algorithms, state, or lifecycles behind a shared interface
     with interchangeable concrete implementations.
4. Check whether the proposed change introduces a new reason for an existing
   class or function to change. If so, create or extend the appropriate
   abstraction instead of mixing responsibilities.

## Architecture

### Preserve Clear Ownership

- Each module, class, and function must have one clear responsibility and a
  small public interface.
- Keep domain knowledge with the object that owns it. Callers should request an
  operation, not inspect internals and reproduce the operation themselves.
- Do not access private members outside their owning class. Add a purposeful
  public method or property when another component needs information.
- Keep orchestration separate from computation. Coordinators assemble and call
  components; domain classes perform the work.
- Pass dependencies explicitly. Avoid mutable module-level state and hidden
  coupling through globals.

### Use Configuration for Selection

- Use the `ConfigBase`, `ConfigField`, `Optional`, and `Choices` framework under
  `ocelot/configs/` for application configuration. Do not add parallel ad hoc
  dictionaries or scattered YAML parsing in model and training code.
- Define and validate each setting in its owning configuration class. Keep one
  source of truth for defaults; do not duplicate defaults at call sites.
- Reject unknown or invalid configuration early with an error that names the
  field and valid choices. Do not silently fall back to a different behavior.
- Update the backing config class, shipped YAML, factory or registry, and tests
  together when adding a configurable component.
- Configuration selects behavior; it must not contain the behavior itself.

### Use Polymorphism for Behavioral Variants

- When implementations differ in algorithm, state, or lifecycle, define a
  narrow common interface and put each implementation in a concrete class.
- Keep type selection in one factory or registry, following the existing
  `make(...)` patterns under `ocelot/model/`. Callers must depend on the common
  interface, not branch on concrete types.
- Adding a new variant should normally require a new config type, concrete
  implementation, and one factory registration. It should not require edits
  throughout the pipeline.
- Do not add chains of `if`/`elif`, `isinstance`, or string comparisons for each
  configured variant outside the construction boundary.
- Put only genuinely shared invariants in base classes. Do not turn a base class
  into a collection of optional hooks and flags for every implementation.
- Prefer composition when features can vary independently. Use inheritance when
  implementations satisfy the same behavioral contract.
- A small stateless transformation may remain a well-named function. Do not add
  a class that has no meaningful state, contract, or ownership merely to claim
  object orientation.

## Readability and Design Rules

- Use descriptive domain names and explicit control flow. Optimize for a human
  reader unfamiliar with the change.
- Keep functions focused. If a function validates configuration, selects an
  implementation, mutates state, performs computation, and formats output,
  separate those responsibilities before extending it.
- Replace repeated branch structures and duplicated algorithms with a shared
  abstraction. Do not hide unrelated operations in a generic helper.
- Avoid boolean mode arguments and long lists of optional parameters that cause
  one function to behave as several different components.
- Use type hints at public boundaries. Make units, shapes, valid ranges, and
  required keys explicit where scientific data enters a component.
- Preserve scientific meaning and numerical behavior during refactors. Do not
  combine domain concepts merely because their arrays have similar shapes.
- Comments and docstrings should explain contracts, invariants, units, or
  non-obvious reasoning. Do not narrate straightforward code.
- Remove dead paths made obsolete by the change. Do not leave commented-out
  implementations or speculative extension points.
- Follow existing style and naming in the local package. Keep changes scoped;
  do not mix unrelated cleanup into a feature or bug fix.

## Warning Signs

Stop and reconsider the design when a change would introduce any of these:

- A growing function with branches for every instrument, mesh, processor,
  sampler, output, or training mode.
- The same configuration discriminator checked in multiple modules.
- Copy-pasted implementations that differ only in a few embedded policies.
- Callers reading private fields or reconstructing another object's derived
  state.
- A class that owns configuration, I/O, domain computation, and presentation.
- Changes to many unrelated call sites just to add one implementation.
- A generic `utils` module becoming the owner of domain-specific behavior.

If one appears, move the behavior to its owning component, extract a policy or
strategy, or extend the relevant factory and interface before proceeding.

## Safe Change Workflow

1. Make the smallest end-to-end change that respects the existing boundaries.
2. Preserve public APIs unless the task explicitly requires a migration.
3. Add or update focused tests for the public behavior, including invalid
   configuration and at least one alternative implementation when relevant.
4. Run the narrowest relevant tests after the first substantive edit. For Python
   changes without targeted tests, run a focused import or compile check for the
   touched package.
5. Review the final diff for duplicated knowledge, leaked private state, new
   variant conditionals, unrelated formatting, and stale configuration.

## Definition of Done

A change is complete when:

- The behavior has one identifiable owner and callers use its public interface.
- Configuration is typed, validated, documented in the shipped YAML when
  applicable, and has a single default.
- A new implementation is selected through the established factory or registry.
- Existing behavior remains covered and the new behavior has focused tests.
- Relevant tests or checks pass, and the final diff contains no unrelated
  refactoring or generated artifacts.