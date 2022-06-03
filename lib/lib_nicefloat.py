import jax


def ja(x):
    """
    Upgrades the storage to jax.numpy.array

    Convenience and shortcut function.

    :param x: float or int
    :return: same float or int, but in jax.numpy.array storage.
    """
    return jax.numpy.array(x)


def execute_backspaces(txt):
    """
    for each "backspace" symbol, '\b' execute the character-killing action here rather than wait for the console to do it.

    for example:

        `python
        txt = '1234\b56\b\b789'
        execute_backspaces(txt) = '123789'
        `

    :param txt: string to trim
    :return: string with backspace action executed.
    """
    pre = txt
    for cnt in range(999):
        pos = pre.find(f'\b')
        if pos == -1:
            break
        post = pre[0:pos - 1] + pre[pos + 1:]
        # print(f'{pre=}, {pos=}, {post=}')
        pre = post
    result = pre
    return result


class NiceFloat:
    """ Beautiful printable floats

    Converts floats, 1D and 2D arrays of floats and integers to their string representation.

    Takes extra care for units, padding, and representing near-zero or near-1 values.

    The concept is for the values to always take exactly 7 characters, enabling nice tabular display.

    """

    def __init__(self, x, suffix='  ', prefix='  ', scale=1e3):
        self.x = x
        self.prefix = prefix
        self.suffix = suffix
        self.scale = scale

    def careful_float_1e6(self, value):
        """
        careful formating for scale = 1e6 (micro-scale)
        :param value: float
        :return: string representation of float, with appropriate suffix.
        """
        return f'{self.prefix}{1e6 * value:5.0f}µ{self.suffix}, '

    def careful_float_1e3(self, value):
        """
        this version does a bit more work when scale = 1e3
        It checks if the result is very small but not excactly zero,
        or close to 1.00 but not exactly 1.00

        :return: string representation of self.x, but with careful handling of nearly-zero and nearly-1
        """
        txt_1st_stage = f'{1e3 * value:5.0f}'
        if value > 0.0 and txt_1st_stage == '    0':
            txt_1st_stage = '   1↘'
        if value < 1.0 and txt_1st_stage == ' 1000':
            txt_1st_stage = ' 999↗'
        txt_2nd_stage = f'{self.prefix}{txt_1st_stage}m{self.suffix}, '
        return txt_2nd_stage

    def careful_float_1(self, value):
        """
        version that converts scale-1 floats to text
        :param value: float
        :return: careful string representation of value
        """
        return f'{self.prefix}{value:5.3f}{self.suffix}, '

    def careful_int(self, value):
        """
        converts integers to text, carefully
        :param value: integer
        :return: careful string representation of value
        """
        return f'{self.prefix}{value: 3d}{self.suffix}, '

    def unknown_scale(self, value):
        """
        displayed when a given float scale is not implemented.
        :param value:
        :return: Ugly error message.
        """
        return f'unknown scale {self.scale} in the initializer. valid values are (1,1e3,1e6)  '

    def __repr__(self):
        x = self.x
        suffix = self.suffix
        prefix = self.prefix
        result = f'unsupported type: {type(x)}'

        # for a python float, treat it like so:
        if isinstance(x, float):
            result = f'{prefix}{jax.numpy.round(x, 2):4.2f}{suffix}'
            return execute_backspaces(result)

        # for a jax array, treat it like this:
        if isinstance(x, jax.interpreters.xla._DeviceArray):
            # decleare the numeric-to-text function, 'txt':

            if x.dtype == jax.numpy.int32:  # if it is an int
                txtf = self.careful_int
            else:  # if it is a float:
                if self.scale == 1e6:
                    txtf = self.careful_float_1e6
                # if scale is 1e3:
                elif self.scale == 1e3:
                    txtf = self.careful_float_1e3
                elif self.scale == 1:
                    txtf = self.careful_float_1
                else:
                    txtf = self.unknown_scale
            if len(x.shape) == 0:
                # scalar. treat as 1D, 1-element array.
                x = ja([x])

            # treat 1D and 2D arrays differently
            if len(x.shape) == 1:
                result = 'jax 1D array'
                if x.shape[0] == 1:
                    result = f'{txtf(x[0])}\b\b'
                    return execute_backspaces(result)
                result = '['
                for element in x:
                    result += txtf(element)
                result += '\b\b]'
                result = execute_backspaces(result)
                return result

            if len(x.shape) == 2:
                result = 'jax 2D array'
                result = ''
                for idx_0 in range(x.shape[0]):
                    if idx_0 == 0:
                        result += '[['  # first
                    else:
                        result += ' ['  # continued
                    for idx_1 in range(x.shape[1]):
                        element = x[idx_0, idx_1]
                        result += txtf(element)
                    if idx_0 != x.shape[0] - 1:
                        result += '\b\b],\n'
                    else:  # final
                        result += '\b\b]]\n'
                result += ''
                result = execute_backspaces(result)
                return result
        # correction: for each "backspace" symbol, execute the action here rather than wait for the console to do it.
        return execute_backspaces(result)

    def __str__(self):
        return execute_backspaces(self.__repr__())

    def __float__(self):
        return float(self.x)

    def __int__(self):
        return int(self.x)


# "printable parameter"
pf = lambda x: NiceFloat(x, prefix='φ=', suffix='  ', scale=1)  # alternate: ϕ

# "printable data"
pd = lambda x: NiceFloat(x, prefix='D=', suffix='  ', scale=1)

# "printable likelihood"
pl = lambda x: NiceFloat(x, suffix='Ω', scale=1e3)

# "printable belief"
pb = lambda x: NiceFloat(x, suffix='R', scale=1e3)

# "printable evidence"
pe = lambda x: NiceFloat(x, suffix='e', scale=1e3)
