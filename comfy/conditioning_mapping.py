
import math


class ConditioningMapping:
    '''
    ConditioningMapping is a helper class to determine the correct mapping between
    conditionings and images in a batch. This allows to have a prompt travel within
    a single batch. A lot of care is taken to ensure that any deviation from the
    default behavior is explicitely requested by the user.
    Fallbacks are in place to ensure that the default behavior is used if any other
    behavior is requested but not possible.
    The class works by accessing the "batch_offset" and "conditioning_mode" fields of the conditionings.
    This needs support from samplers.py

    The field "conditioning_mode" requests a specific strategy for the conditionings.
    The field "batch_offset" specifies the offset into the image batch for the conditioning.
    '''

    MODE_KEY = "conditioning_mode"
    OFFSET_KEY = "batch_offset"

    POLARITY_POSITIVE = "positive"
    POLARITY_NEGATIVE = "negative"

    MODE_DENSE = "dense_conditioning"
    '''
    Dense strategy: all conditionings are applied to all images in the batch.
    This means for a batch of size 32 and 10 conditionings, the cross attention
    will be evaluated 320 times per sampling step.
    This is the default behavior of comfy and the only one that works with controlnets.
    If there is no explicit strategy specified, this strategy is used.
    If there is any issue with determining the correct strategy, this strategy is used (with warning).
    If there is any controlnet in a conditioning, this strategy is used (with warning).
    '''

    MODE_EXPLICIT = "explicit_conditioning"
    '''
    Each conditioning has a field `batch_offset` which specifies the offset into the
    image batch. A conditioning with this mode only affects the single image at the
    specified offset.
    WARNING: Does not yet work with controlnets.
    '''

    MODE_BLOCK_DIAGONAL = "block_diagonal_conditioning"
    '''
    Block diagonal strategy: each image in the batch is only conditioned by a subset
    of the conditionings.
    If there are 8 images in the batch and 32 conditionings, each image will be
    conditioned by 4 conditionings.
    This allows to have a prompt travel within a single batch.
    '''

    def auto_set_conditionings(image_count: int, positives: list, negatives: list = []) -> (str, str):
        '''
        Determines the correct strategy for the given conditionings.
        If dense strategy is determined, the batch_offset field of all conditionings
        is removed to signal that the offset is not used.
        In all other cases, the batch_offset field is set to the correct offset.
        The polarity field is only used to print better warnings.
        If there is a controlnet in positives, also the negatives have to fall back to dense, thus both
        have to be analyzed at the same time.
        The return value is a tuple of the determined strategies for the positive and negative conditionings.
        If the default (dense) strategy is used, None is returned. However, if the default strategy is forced
        by a controlnet, the default strategy is returned explicitly.
        '''

        if negatives is None:
            negatives = []

        # determine the strategy for the positive conditionings
        positive_strategy = ConditioningMapping.detect_strategy(
            positives, ConditioningMapping.POLARITY_POSITIVE, image_count)
        
        # determine the strategy for the negative conditionings
        negative_strategy = ConditioningMapping.detect_strategy(
            negatives, ConditioningMapping.POLARITY_NEGATIVE, image_count)
        
        forced_fallback = False
        if positive_strategy == ConditioningMapping.MODE_DENSE:
            forced_fallback = True
        if negative_strategy == ConditioningMapping.MODE_DENSE:
            forced_fallback = True

        if forced_fallback:
            positive_strategy = ConditioningMapping.MODE_DENSE
            negative_strategy = ConditioningMapping.MODE_DENSE

        pos_res = positive_strategy
        neg_res = negative_strategy

        if positive_strategy is None:
            positive_strategy = ConditioningMapping.MODE_DENSE

        if negative_strategy is None:
            negative_strategy = ConditioningMapping.MODE_DENSE

        ConditioningMapping.set_strategy(
            positives, image_count, positive_strategy)
        
        ConditioningMapping.set_strategy(
            negatives, image_count, negative_strategy)
        
        # return None for each if the default strategy (dense) is used but not forced.
        return (pos_res, neg_res)


    def has_any_controlnet(conditionings: list) -> bool:
        return any("control" in c[1] for c in conditionings)

    def has_any_block_diag_request(conditionings: list) -> bool:
        # look if there is a MODE_KEY with MODE_BLOCK_DIAGONAL in any conditioning
        for ct in conditionings:
            c = ct[1]
            if ConditioningMapping.MODE_KEY in c:
                if c[ConditioningMapping.MODE_KEY] == ConditioningMapping.MODE_BLOCK_DIAGONAL:
                    return True
        return False

    def has_any_dense_request(conditionings: list) -> bool:
        # look if there is a MODE_KEY with MODE_DENSE in any conditioning
        for ct in conditionings:
            c = ct[1]
            if ConditioningMapping.MODE_KEY in c:
                if c[ConditioningMapping.MODE_KEY] == ConditioningMapping.MODE_DENSE:
                    return True
        return False

    def has_any_explicit_request(conditionings: list) -> bool:
        # look for any batch_offset >= 0
        for ct in conditionings:
            c = ct[1]
            if ConditioningMapping.OFFSET_KEY in c:
                if c[ConditioningMapping.OFFSET_KEY] >= 0:
                    return True
        return False

    def has_all_valid_explicit_offset(conditionings: list, image_count: int) -> bool:
        for ct in conditionings:
            c = ct[1]
            if ConditioningMapping.OFFSET_KEY in c:
                offset = c[ConditioningMapping.OFFSET_KEY]
                if offset < 0:
                    return False  # negative offset is not valid for explicit mode
                if offset >= image_count:
                    return False  # offset larger than image count is not valid
            else:
                return False  # missing offset is not valid for explicit mode

        return True

    def set_strategy(conditionings: list, image_count: int, mode: str):
        '''
        Changes the strategy of the conditionings to the given mode.
        If mode is MODE_DENSE, it will remove the batch_offset and mode fields
        from all conditionings, resetting everything to the default behavior.
        For MODE_EXPLICIT, it will set the mode field to MODE_EXPLICIT and
        validate all batch_offset fields.
        In MODE_BLOCK_DIAGONAL, it will set the mode field to MODE_BLOCK_DIAGONAL
        and attempt to compute the correct batch_offset fields.
        If an error is encountered, an exception is thrown.
        Use auto_set_conditionings for automatic handling of the mode.
        '''
        if mode == ConditioningMapping.MODE_DENSE:
            for ct in conditionings:
                c = ct[1]
                # remove the batch_offset key
                if ConditioningMapping.OFFSET_KEY in c:
                    del c[ConditioningMapping.OFFSET_KEY]

                # remove the mode key
                if ConditioningMapping.MODE_KEY in c:
                    del c[ConditioningMapping.MODE_KEY]

        elif mode == ConditioningMapping.MODE_EXPLICIT:
            for ct in conditionings:
                c = ct[1]
                # validate all offset fields

                # ensure that the offset is present and in the correct range
                if ConditioningMapping.OFFSET_KEY not in c:
                    raise Exception(
                        "Explicit conditioning mapping was requested, but there is a conditioning without batch_offset field")

                offset = c[ConditioningMapping.OFFSET_KEY]

                if offset < 0:
                    raise Exception(
                        "Explicit conditioning mapping was requested, but there is a negative batch_offset field")

                if offset >= image_count:
                    raise Exception(
                        "Explicit conditioning mapping was requested, but there is a batch_offset field that is larger than the image count")

                # set the mode key to explicit
                c[ConditioningMapping.MODE_KEY] = ConditioningMapping.MODE_EXPLICIT

        elif mode == ConditioningMapping.MODE_BLOCK_DIAGONAL:
            # check if there are more images than conditionings
            if image_count > len(conditionings):
                raise Exception(
                    "Block diagonal conditioning requested, but there are more images than conditionings.")

            # check if the conditioning count is divisible by the image count
            if len(conditionings) % image_count != 0:
                raise Exception(
                    "Block diagonal conditioning requested, but the conditioning count is not divisible by the image count.")

            # set the mode key to block diagonal
            for ct in conditionings:
                c = ct[1]
                c[ConditioningMapping.MODE_KEY] = ConditioningMapping.MODE_BLOCK_DIAGONAL

            # set the batch_offset key to the correct offset
            for i, ct in enumerate(conditionings):
                c = ct[1]
                offset = i % image_count
                c[ConditioningMapping.OFFSET_KEY] = offset

    def detect_strategy(conditionings: list, polarity: str, image_count: int) -> str | None:
        """
        Detects the strategy for a list of conditionings.
        Dense strategy is the default that should be used if the detected strategy is None.
        Dense strategy WILL be returned, if no other strategy is possible also for the other polarity.
        """

        if len(conditionings) <= 1:
            return None  # use default mode: dense for this polarity, no info on the other polarity

        dense_requested = ConditioningMapping.has_any_dense_request(
            conditionings)
        block_diag_requested = ConditioningMapping.has_any_block_diag_request(
            conditionings)
        explicit_requested = ConditioningMapping.has_any_explicit_request(
            conditionings)
        controlnet_present = ConditioningMapping.has_any_controlnet(
            conditionings)

        all = [dense_requested, block_diag_requested, explicit_requested]
        if all.count(True) > 1:
            raise Exception(
                f"Multiple different conditioning strategies explicitely requested for {polarity}. Cannot be satisfied.")

        if dense_requested:
            return None  # use default mode: dense for this polarity, no info on the other polarity

        if controlnet_present and all.count(True) > 0:
            print(
                f"WARNING: Controlnet present in {polarity} conditioning. Falling back to default conditioning.")
            # report dense explicitly if the decision is forced by a controlnet
            return ConditioningMapping.MODE_DENSE

        if explicit_requested:
            # validate all offset fields
            if ConditioningMapping.has_all_valid_explicit_offset(conditionings, image_count):
                return ConditioningMapping.MODE_EXPLICIT
            else:
                print(
                    f"WARNING: Invalid batch_offset in {polarity} conditioning. Falling back to default conditioning.")
                return None  # use default mode: dense for this polarity, no info on the other polarity

        if not block_diag_requested:
            return None  # use default mode: dense for this polarity, no info on the other polarity

        # block diagonal requested, now we need to check if it is possible
        n_conditionings = len(conditionings)

        if image_count > n_conditionings:
            print(
                f"WARNING: More images than conditionings for {polarity} conditioning. Falling back to default conditioning.")
            return None  # use default mode: dense for this polarity, no info on the other polarity

        if n_conditionings % image_count != 0:
            print(
                f"WARNING: Conditionings not divisible by images for {polarity} conditioning. Falling back to default conditioning.")
            return None  # use default mode: dense for this polarity, no info on the other polarity

        return ConditioningMapping.MODE_BLOCK_DIAGONAL
    
    def chunk_conditionings(image_count: int, chunk_size: int, positives: list, negatives: list = []) -> list[tuple[list, list]]:
        '''
        Chunks the conditionings into chunks of chunk_size. The last chunk may be smaller.
        The chunks are returned as a list of tuples.
        Each tuples contains a list of positive and a list of negative conditionings.
        The goal is to create the chunks in a way that conditions the images in exactly the
        same way as if the whole batch was processed at once.
        '''

        (pos_mode, neg_mode) = ConditioningMapping.auto_set_conditionings(
            image_count, positives, negatives)

        result = []

        n_chunks = math.ceil(image_count / chunk_size)
        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i+1) * chunk_size, image_count)

            pos = []
            neg = []

            # if no mode is detected or dense mode is forced,
            # we can just copy the whole list for each chunk.
            # In the other modes, we need to change the offsets to match the chunk.
            # We can also remove the conditionings that affect only other chunks.

            if pos_mode is None or pos_mode == ConditioningMapping.MODE_DENSE:
                pos = positives.copy()
            else:
                for ct in positives:
                    c = ct[1]
                    offset = c[ConditioningMapping.OFFSET_KEY]
                    if offset >= start and offset < end:
                        c[ConditioningMapping.OFFSET_KEY] = offset - start
                        pos.append(ct)

            if neg_mode is None or neg_mode == ConditioningMapping.MODE_DENSE:
                neg = negatives.copy()
            else:
                for ct in negatives:
                    c = ct[1]
                    offset = c[ConditioningMapping.OFFSET_KEY]
                    if offset >= start and offset < end:
                        c[ConditioningMapping.OFFSET_KEY] = offset - start
                        neg.append(ct)

            result.append((pos, neg))
        
        return result
    
    def copy_conditioning_list(conditionings: list) -> list:
        '''
        Returns a copy of the given list of conditionings.
        The copy is only for the metadata, so the conditionings themselves are not copied.
        '''
        output = []

        for ct in conditionings:
            old_embedding = ct[0]
            old_metadata = ct[1]
            new_metadata = {}

            for k in old_metadata:
                # make a copy of each key/value, but not a deep copy
                v = old_metadata[k]
                new_metadata[k] = v

            output.append((old_embedding, new_metadata))

        return output

