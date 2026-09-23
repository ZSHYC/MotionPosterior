from collections import deque


def iter_sliding_windows(cap, num_frames, mode):
    """Yield ``(frames, output_position, frame_index)`` for true sliding inference."""
    if mode not in ('center', 'causal'):
        raise ValueError(f"unsupported sliding mode: {mode}")
    ok, first = cap.read()
    if not ok:
        return
    window = deque(maxlen=num_frames)
    if mode == 'center':
        radius = num_frames // 2
        for _ in range(radius + 1):
            window.append(first)
        total_read = 1
        eof = False
        for _ in range(radius):
            ok, frame = cap.read()
            if ok:
                window.append(frame)
                total_read += 1
            else:
                eof = True
                window.append(window[-1])
        output_position = radius
    else:
        for _ in range(num_frames):
            window.append(first)
        total_read = 1
        eof = False
        output_position = num_frames - 1

    frame_index = 0
    while frame_index < total_read:
        yield list(window), output_position, frame_index
        frame_index += 1
        window.popleft()
        if not eof:
            ok, frame = cap.read()
            if ok:
                window.append(frame)
                total_read += 1
            else:
                eof = True
                window.append(window[-1])
        else:
            window.append(window[-1])
