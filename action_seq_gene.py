import time

# Sample hand event data
hand_events = {
    'close': [
        (0.6563838578405834, 0.27109536883376895),
        (0.625005148705982, 0.41621464490890503),
        (0.489688074304944, 0.2216957892690386),
        (0.5892637286867414, 0.46340099828583853),
        (0.336188588823591, 0.2625746095464343)
    ],
    'open': [
        (0.6288775148845854, 0.40165958801905316),
        (0.6193806245213463, 0.17991815578369869),
        (0.6085330758775983, 0.3984893134662083),
        (0.5117876756758917, 0.37426034041813444),
        (0.37431590188117253, 0.2420923440229325)
    ]
}



def generate_robot_action_sequence(hand_events):
    """Generates the robot arm's action sequence based on hand events."""
    
    # Iterate over the hand events and generate actions
    sequence = []
    
    # Make sure the hand_events have an equal number of "close" and "open" events
    max_len = max(len(hand_events['close']), len(hand_events['open']))
    for i in range(max_len):
        if i < len(hand_events['close']):
            close_x, close_y = hand_events['close'][i]
            sequence.append(f"Move to grasp position ({close_x}, {close_y})")
            sequence.append("Grasp")
         
            
        if i < len(hand_events['open']):
            open_x, open_y = hand_events['open'][i]
            sequence.append(f"Move to release position ({open_x}, {open_y})")
            sequence.append('Release')
   

    return sequence

# Generate and print the action sequence
action_sequence = generate_robot_action_sequence(hand_events)
for action in action_sequence:
    print(action)
