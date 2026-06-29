import re
import buddy
import spot
from itertools import product


def extract_atomic_props(formula):
    tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', formula)
    keywords = {
        'X', 'G', 'F', 'U', 'R', 'W', 'M', '&&', '||', '!', 'true', 'false',
        'True', 'False', 'not', 'and', 'or',
    }
    atomic_props = sorted(set(tok for tok in tokens if tok not in keywords))
    return atomic_props


def extract_dfa_transitions_with_trash_expanded(formula):
    dfa = spot.translate(formula, 'deterministic', 'complete')
    bdd_dict = dfa.get_dict()
    num_states = dfa.num_states()

    is_sink = [False] * num_states
    trash_states_set = set()
    for s in range(num_states):
        outgoing = list(dfa.out(s))
        if len(outgoing) == 1:
            tr = outgoing[0]
            if tr.dst == s and tr.cond == buddy.bddtrue and not dfa.state_is_accepting(s):
                is_sink[s] = True
                trash_states_set.add(s)

    state_names = {}
    for i in range(num_states):
        if dfa.state_is_accepting(i):
            state_names[i] = "accept_all"
        elif is_sink[i]:
            state_names[i] = "Trash"
        else:
            state_names[i] = str(i)

    initial_state_index = dfa.get_init_state_number()
    initial_state_name = state_names[initial_state_index]
    print(f"Initial state index: {initial_state_index}")
    print(f"Initial state name: {initial_state_name}")

    atomic_props = [str(ap) for ap in dfa.ap()]
    all_valuations = list(product([False, True], repeat=len(atomic_props)))

    def valuation_to_formula(valuation):
        return ' && '.join([prop if val else f'!{prop}' for prop, val in zip(atomic_props, valuation)])

    def valuation_satisfies(cond_bdd, valuation):
        val_bdd = buddy.bddtrue
        for prop, val in zip(atomic_props, valuation):
            var_num = bdd_dict.varnum(prop)
            var_bdd = buddy.bdd_ithvar(var_num)
            if not val:
                var_bdd = buddy.bdd_not(var_bdd)
            val_bdd = buddy.bdd_and(val_bdd, var_bdd)
        product_bdd = buddy.bdd_and(cond_bdd, val_bdd)
        return product_bdd != buddy.bddfalse

    expanded_transitions = []
    for s in range(num_states):
        for tr in dfa.out(s):
            src = state_names[s]
            dst = state_names[tr.dst]
            cond_bdd = tr.cond

            for valuation in all_valuations:
                if valuation_satisfies(cond_bdd, valuation):
                    # cond_str = valuation_to_formula(valuation)
                    # expanded_transitions.append((src, [cond_str], dst))
                    cond_str = normalize_condition(valuation_to_formula(valuation))
                    expanded_transitions.append((src, [cond_str], dst))

    return expanded_transitions, initial_state_name, trash_states_set


def normalize_condition(condition):
    parts = condition.split(" && ")
    sorted_parts = sorted(parts)
    normalized_condition = " && ".join(sorted_parts)
    return normalized_condition


def is_condition_covered(target_cond, existing_conditions):
    target_cond_parts = set(normalize_condition(target_cond).split(" && "))

    if isinstance(existing_conditions, str):
        existing_conditions = [existing_conditions]

    for existing_cond in existing_conditions:
        existing_cond_parts = set(normalize_condition(existing_cond).split(" && "))
        if target_cond_parts <= existing_cond_parts or existing_cond_parts.issubset(target_cond_parts):
            return True

    return False


def probabilistic_labeling_next(transitions, observation_probabilities, dfa_transitions, adj_matrix):
    transition_dict = {}
    for transition in transitions:
        transition_dict[(repr(transition[0]), repr(transition[1]))] = 0
        part_1 = transition[0][1]
        part_2 = transition[1][1]

        for dfa_transition in dfa_transitions:
            if dfa_transition[0] == part_1 and dfa_transition[2] == part_2:
                label = dfa_transition[1][0]

                # for value, obs in observation_probabilities[transition[1][0]]:
                #     if obs == label:
                #         transition_dict[(repr(transition[0]), repr(transition[1]))] += value
                for value, obs in observation_probabilities[transition[1][0]]:
                    if normalize_condition(obs) == normalize_condition(label):
                        transition_dict[(repr(transition[0]), repr(transition[1]))] += value

                if part_1 == 'accept_all' and part_2 == 'accept_all':
                    transition_dict[(repr(transition[0]), repr(transition[1]))] = 1
                if part_1 == 'Trash' and part_2 == 'Trash':
                    transition_dict[(repr(transition[0]), repr(transition[1]))] = 1

    return transition_dict
