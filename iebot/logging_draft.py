def iebot_v4(obs, config):
    import logging

    board = translate_board(obs.board)
    bit_board, mask = get_position_mask_bitmap(board=board, player=2 if obs.mark == 1 else 1)
    columns_map = generate_columns_map(mask)

    node = Node(bit_board, mask)
    # TODO rewrite this to make more elegant the use of a list here
    _, play = negamax_ab(node=node, depth=5, columns_map=columns_map)

    debug_msg = "bit board: {0} mask {1} result {2} play {3}".format(bit_board, mask, _, play)
    logging.basicConfig(level=logging.INFO,
                        filemode='a',
                        filename="my.log",
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S')
    logging.critical(debug_msg)

    return transform_play_to_column(play=play, columns_map=columns_map)