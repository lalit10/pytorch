# Owner(s): ["module: dynamo"]

import cmath
import decimal
import fractions
import gc
import math
import operator
import random
from unittest import mock

import torch
from torch._dynamo.exc import Unsupported
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import CompileCounter
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)


NAN_IDENTITY_OPS = {
    "dict_get": lambda a, b: {a: 1}.get(b, "missing"),
    "set_contains": lambda a, b: b in {a},
    "list_contains": lambda a, b: b in [a],
    "list_eq": lambda a, b: [a] == [b],
    "tuple_eq": lambda a, b: (a,) == (b,),
    "list_count": lambda a, b: [a].count(b),
    "is": lambda a, b: a is b,
}


class DecimalTests(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_lazy_wrapper_identity(self):
        from torch._dynamo.source import LocalSource
        from torch._dynamo.variables.decimal import DecimalVariable
        from torch._dynamo.variables.lazy import LazyCache, LazyVariableTracker
        from torch._dynamo.variables.object_protocol import vt_identity_compare

        value = decimal.Decimal("NaN")
        source = LocalSource("k")
        inner = DecimalVariable(value, source=source)
        cache = LazyCache(value, source)
        cache.vt = inner
        wrapper = LazyVariableTracker(cache, source=source)
        other_wrapper = LazyVariableTracker(cache, source=source)
        unrealized = LazyVariableTracker(LazyCache(value, source), source=source)
        with (
            mock.patch.object(
                LazyVariableTracker, "realize", side_effect=AssertionError
            ),
            mock.patch.object(
                DecimalVariable,
                "get_real_python_backed_value",
                side_effect=AssertionError,
            ),
        ):
            for left, right in (
                (wrapper, inner),
                (inner, wrapper),
                (wrapper, other_wrapper),
                (unrealized, unrealized),
            ):
                self.assertTrue(vt_identity_compare(left, right).as_python_constant())
        self.assertFalse(unrealized.is_realized())

    def test_nan_key_lookup(self):
        def fn(x, k):
            try:
                return x + 1, {k: 1}[k], {k: 2}.get(k, "missing")
            except KeyError:
                return x + 1, "missing", "missing"

        x = torch.ones(2)
        k = decimal.Decimal("NaN")
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(x, k), fn(x, k))

    @parametrize("name", list(NAN_IDENTITY_OPS))
    @parametrize("first", ["same", "distinct"])
    def test_nan_input_identity(self, name, first):
        op = NAN_IDENTITY_OPS[name]

        def fn(x, a, b):
            return x + 1, op(a, b)

        n1, n2 = decimal.Decimal("NaN"), decimal.Decimal("NaN")
        same, distinct = (n1, n1), (n1, n2)
        order = [same, distinct] if first == "same" else [distinct, same]
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        # Each relation compiles once; repeating the sequence hits the cache.
        for args in order * 2:
            self.assertEqual(compiled(x, *args), fn(x, *args))
        self.assertEqual(counter.frame_count, 2)

    def test_equal_non_nan_inputs_share_graph(self):
        def fn(x, a, b):
            return x + 1, b in [a], {a: 1}.get(b)

        d1, d2 = decimal.Decimal("1.5"), decimal.Decimal("1.5")
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        for args in ((d1, d1), (d1, d2)):
            self.assertEqual(compiled(x, *args), fn(x, *args))
        self.assertEqual(counter.frame_count, 1)

    @torch._dynamo.config.patch(recompile_limit=64)
    def test_nan_identity_and_values_after_collection(self):
        def fn(x, a, b):
            return x + 1, b in [a], a.as_tuple(), b.as_tuple(), str(a), str(b)

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        values = ("NaN1", "NaN2", "1.5", "-0", "Infinity")
        for i in range(12):
            a = decimal.Decimal(values[i % len(values)])
            b = a if i % 2 else decimal.Decimal(values[i % len(values)])
            self.assertEqual(compiled(x, a, b), fn(x, a, b))
            del a, b
            gc.collect()
        self.assertGreater(counter.frame_count, 0)

    def test_nan_input_vs_local_and_constructor_alias(self):
        def fn(x, k):
            alias = decimal.Decimal(k)
            local = decimal.Decimal("NaN")
            copied = decimal.Context().create_decimal(k)
            return x + 1, (
                alias is k,
                alias in [k],
                local is k,
                local in [k],
                copied is k,
                copied in [k],
            )

        x = torch.ones(2)
        k = decimal.Decimal("NaN")
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        for _ in range(2):
            self.assertEqual(compiled(x, k), fn(x, k))

    def test_same_signaling_nan_container_identity(self):
        def fn(x, k):
            return x + 1, k in [k], [k].count(k), [k] == [k], k is k

        x = torch.ones(2)
        k = decimal.Decimal("sNaN")
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        with decimal.localcontext() as ambient:
            ambient.traps[decimal.InvalidOperation] = True
            ambient.clear_flags()
            before = str(ambient)
            for _ in range(2):
                self.assertEqual(compiled(x, k), fn(x, k))
                self.assertEqual(str(ambient), before)

    def test_distinct_signaling_nan_container_fallback(self):
        def fn(x, a, b):
            try:
                result = b in [a]
            except decimal.InvalidOperation:
                result = "trapped"
            return x + 1, result

        x = torch.ones(2)
        a, b = decimal.Decimal("sNaN"), decimal.Decimal("sNaN")
        with decimal.localcontext() as ambient:
            ambient.clear_flags()
            before = str(ambient)
            with self.assertRaises(Unsupported):
                torch.compile(fn, backend="eager", fullgraph=True)(x, a, b)
            self.assertEqual(str(ambient), before)
            compiled = torch.compile(fn, backend="eager", fullgraph=False)
            for trapped in (False, True, False):
                ambient.traps[decimal.InvalidOperation] = trapped
                ambient.clear_flags()
                expected = fn(x, a, b)
                expected_flags = dict(ambient.flags)
                ambient.clear_flags()
                self.assertEqual(compiled(x, a, b), expected)
                self.assertEqual(dict(ambient.flags), expected_flags)

    def test_equal_valued_inputs_reuse_graph(self):
        def fn(x, d):
            return x + float(d), str(d), d

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        inputs = [decimal.Decimal("7.25") for _ in range(4)]
        x = torch.ones(2)
        for d in inputs:
            actual = compiled(x, d)
            self.assertEqual(actual[:2], fn(x, d)[:2])
            self.assertIs(actual[2], d)
        self.assertEqual(counter.frame_count, 1)

    @torch._dynamo.config.patch(recompile_limit=64)
    def test_fresh_inputs_after_collection(self):
        def fn(x, d):
            return x + float(d), float(d), str(d)

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        for i in range(12):
            d = decimal.Decimal(f"{i}.5")
            self.assertEqual(compiled(x, d), fn(x, d))
            del d
            gc.collect()
        self.assertEqual(counter.frame_count, 12)

    @torch._dynamo.config.patch(recompile_limit=64)
    def test_input_representation_guard(self):
        def fn(x, d):
            return x + 1, str(d), d.as_tuple(), d

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        values = [
            "0",
            "-0",
            "0E-100",
            "1.0",
            "1.00",
            "Infinity",
            "-Infinity",
            "NaN123",
            "sNaN123",
        ]
        for value in values:
            d = decimal.Decimal(value)
            actual = compiled(x, d)
            self.assertEqual(actual[:3], fn(x, d)[:3])
            self.assertIs(actual[3], d)
        self.assertEqual(counter.frame_count, len(values))

    @torch._dynamo.config.patch(recompile_limit=64)
    def test_fresh_inputs_comparison_and_hash(self):
        def fn(x, a, b):
            return x + 1, a == b, a < b, hash(a)

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        for i in range(12):
            a = decimal.Decimal(f"{i}.5")
            b = decimal.Decimal(f"{(i * 7) % 5}.5")
            self.assertEqual(compiled(x, a, b), fn(x, a, b))
            del a, b
            gc.collect()
        self.assertEqual(counter.frame_count, 12)

    @parametrize("value", ["-0", "1.00", "1E+100", "Infinity", "-Infinity", "sNaN123"])
    def test_hash(self, value):
        def fn(x, d):
            try:
                return x + 1, hash(d), d.__hash__()
            except TypeError:
                return x + 2, None

        x = torch.ones(2)
        d = decimal.Decimal(value)
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x, d), fn(x, d)
        )

    def test_quiet_nan_input_hash(self):
        def fn(x, d):
            return x + 1, hash(d), d.__hash__(), d

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        inputs = [decimal.Decimal("NaN123") for _ in range(2)]
        for d in (*inputs, *inputs):
            actual = compiled(x, d)
            self.assertEqual(actual[:3], fn(x, d)[:3])
            self.assertIs(actual[3], d)
        self.assertEqual(counter.frame_count, 2)

    @parametrize(
        "op",
        [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge],
    )
    @parametrize("other", [decimal.Decimal("1.5"), 1, True, "1.5", 1 + 2j])
    def test_comparisons(self, op, other):
        def fn(x, d):
            try:
                return x + 1, op(d, other), op(other, d)
            except TypeError:
                return x + 2, None

        x = torch.ones(2)
        d = decimal.Decimal("1.5")
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x, d), fn(x, d)
        )

    def test_decimal_dict_set_and_sort(self):
        def fn(x, a, b, equal):
            table = {a: 1, b: 2}
            values = {a, b}
            return x + table[equal], equal in values, sorted([b, a]), len(values)

        x = torch.ones(2)
        a, b, equal = (decimal.Decimal(value) for value in ("1.0", "2.0", "1.00"))
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x, a, b, equal),
            fn(x, a, b, equal),
        )

    def test_input_alias_guards(self):
        def fn(x, a, b):
            return x + 1, a is b, a is not b, a, b

        compiled = torch.compile(fn, backend="eager", fullgraph=False)
        x = torch.ones(2)
        a, b = (decimal.Decimal("1.5") for _ in range(2))
        for left, right in ((a, a), (a, b), (b, a), (b, b)):
            actual = compiled(x, left, right)
            self.assertEqual(actual[:3], fn(x, left, right)[:3])
            self.assertIs(actual[3], left)
            self.assertIs(actual[4], right)

    def test_comparison_slots_not_implemented(self):
        def fn(x, d):
            return x + 1, d.__eq__("1"), d.__ne__(None), d.__lt__(b"1")

        x = torch.ones(2)
        d = decimal.Decimal(1)
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x, d), fn(x, d)
        )

    def test_comparison_integer_subclass(self):
        class Integer(int):
            pass

        def fn(x, d, other):
            return x + 1, d == other, d != other, d < other

        x = torch.ones(2)
        d = decimal.Decimal(1)
        compiled = torch.compile(fn, backend="eager", fullgraph=False)
        for value in (0, 1, 2):
            other = Integer(value)
            self.assertEqual(compiled(x, d, other), fn(x, d, other))

    def test_comparison_reflected_method(self):
        class Other:
            def __eq__(self, other):
                return "reflected eq"

            def __gt__(self, other):
                return "reflected gt"

        def fn(x, d, other):
            return x + 1, d == other, d < other

        x = torch.ones(2)
        d = decimal.Decimal(1)
        other = Other()
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=False)(x, d, other),
            fn(x, d, other),
        )

    def test_comparison_fraction(self):
        def fn(x, d, other):
            return x + 1, d == other, d != other, d < other

        x = torch.ones(2)
        d = decimal.Decimal("0.5")
        compiled = torch.compile(fn, backend="eager", fullgraph=False)
        for other in (
            fractions.Fraction(1, 4),
            fractions.Fraction(1, 2),
            fractions.Fraction(3, 4),
        ):
            self.assertEqual(compiled(x, d, other), fn(x, d, other))

    def test_comparison_fraction_subclass(self):
        class Rational(fractions.Fraction):
            def __eq__(self, other):
                return False

            def __gt__(self, other):
                return False

        def fn(x, d, other):
            return x + 1, d == other, d < other

        x = torch.ones(2)
        d = decimal.Decimal("0.5")
        compiled = torch.compile(fn, backend="eager", fullgraph=False)
        for other in (Rational(1, 2), Rational(3, 4)):
            self.assertEqual(compiled(x, d, other), fn(x, d, other))

    def test_sourceless_quiet_nan_hash_does_not_escape(self):
        def fn(x):
            d = decimal.Decimal("NaN123")
            return x + 1, d, hash(d)

        x = torch.ones(2)
        with self.assertRaises(Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(x)
        compiled = torch.compile(fn, backend="eager", fullgraph=False)
        for _ in range(3):
            actual = compiled(x)
            self.assertEqual(actual[0], x + 1)
            self.assertEqual(actual[1].as_tuple(), decimal.Decimal("NaN123").as_tuple())
            self.assertEqual(actual[2], hash(actual[1]))

    def test_sourceless_quiet_nan_dict_keys(self):
        def fn(x):
            a = decimal.Decimal("NaN")
            b = decimal.Decimal("NaN")
            table = {a: 1, b: 2}
            return x + table[a], table[b], len(table), a == b

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize(
        "op",
        [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge],
    )
    @parametrize(
        "case",
        [
            ("1.5", 1.5, decimal.FloatOperation),
            ("1.5", 1.5 + 0j, decimal.FloatOperation),
            ("NaN123", decimal.Decimal(1), decimal.InvalidOperation),
            ("sNaN123", decimal.Decimal(1), decimal.InvalidOperation),
        ],
    )
    @parametrize("trapped", [False, True])
    def test_localcontext_comparison_flags_and_traps(self, op, case, trapped):
        value, other, signal = case

        def fn(x, d):
            with decimal.localcontext(
                decimal.Context(traps=[signal] if trapped else [])
            ) as context:
                try:
                    result = op(d, other)
                except (decimal.DecimalException, TypeError):
                    result = "caught"
            return x + 1, result, context

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        d = decimal.Decimal(value)
        with decimal.localcontext() as ambient:
            before = str(ambient)
            expected = fn(x, d)
            for _ in range(2):
                actual = compiled(x, d)
                self.assertEqual(actual[:2], expected[:2])
                self.assertEqual(str(actual[2]), str(expected[2]))
                self.assertEqual(str(ambient), before)
        self.assertEqual(counter.frame_count, 1)

    @parametrize(
        "case",
        [
            (operator.eq, "1.5", 1.5),
            (operator.lt, "1.5", 1.5),
            (operator.eq, "1.5", 1.5 + 0j),
            (operator.lt, "NaN123", decimal.Decimal(1)),
            (operator.eq, "sNaN123", decimal.Decimal(1)),
        ],
    )
    def test_ambient_comparison_does_not_leak(self, case):
        op, value, other = case

        def fn(x, d):
            return x + 1, op(d, other)

        with decimal.localcontext() as ambient:
            ambient.traps[decimal.FloatOperation] = False
            ambient.traps[decimal.InvalidOperation] = False
            ambient.clear_flags()
            before = str(ambient)
            with self.assertRaises(Unsupported):
                torch.compile(fn, backend="eager", fullgraph=True)(
                    torch.ones(2), decimal.Decimal(value)
                )
            self.assertEqual(str(ambient), before)

    @parametrize(
        "case",
        [
            (operator.eq, "1.5", 1.5),
            (operator.lt, "1.5", 1.5),
            (operator.eq, "1.5", 1.5 + 0j),
            (operator.lt, "NaN123", decimal.Decimal(1)),
            (operator.eq, "sNaN123", decimal.Decimal(1)),
        ],
    )
    def test_ambient_comparison_fallback_flags_and_traps(self, case):
        op, value, other = case

        def fn(x, d):
            try:
                result = op(d, other)
            except decimal.DecimalException:
                result = "caught"
            return x + 1, result

        compiled = torch.compile(fn, backend="eager", fullgraph=False)
        x = torch.ones(2)
        d = decimal.Decimal(value)
        with decimal.localcontext() as ambient:
            for enabled in (False, True, False):
                ambient.traps[decimal.FloatOperation] = enabled
                ambient.traps[decimal.InvalidOperation] = enabled
                ambient.clear_flags()
                expected = fn(x, d)
                expected_flags = dict(ambient.flags)
                ambient.clear_flags()
                self.assertEqual(compiled(x, d), expected)
                self.assertEqual(dict(ambient.flags), expected_flags)

    @parametrize(
        "value",
        [
            "1.2500",
            "-0",
            "0E-100",
            "1E+100",
            "Infinity",
            "-Infinity",
            "NaN123",
            "sNaN42",
            0,
            123,
            (1, (1, 2, 3), -2),
        ],
    )
    def test_construction_and_reconstruction(self, value):
        def fn(x):
            d = decimal.Decimal(value)
            return x + 1, d, d, d.as_tuple()

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        for _ in range(2):
            result = compiled(torch.ones(2))
            self.assertEqual(result[0], torch.full((2,), 2.0))
            self.assertIs(type(result[1]), decimal.Decimal)
            self.assertIs(result[1], result[2])
            self.assertEqual(result[1].as_tuple(), decimal.Decimal(value).as_tuple())
            self.assertEqual(result[3], decimal.Decimal(value).as_tuple())
        self.assertEqual(counter.frame_count, 1)

    def test_constructor_signature(self):
        def fn(x):
            a = decimal.Decimal()
            b = decimal.Decimal(value="1.25", context=None)
            c = decimal.Decimal("2.5", None)
            return x + float(a) + float(b) + float(c)

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    def test_input_identity_and_guard(self):
        def fn(x, d):
            return x + float(d), d, decimal.Decimal(d)

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        for value in ("1.25", "2.5"):
            d = decimal.Decimal(value)
            for _ in range(2):
                result = compiled(torch.ones(2), d)
                self.assertEqual(result[0], torch.ones(2) + float(d))
                self.assertIs(result[1], d)
                self.assertIs(result[2], d)
        self.assertEqual(counter.frame_count, 2)

    def test_constructor_identity_with_context_and_keywords(self):
        def fn(x, d):
            context = decimal.Context()
            a = decimal.Decimal(value=d)
            b = decimal.Decimal(d, context)
            with decimal.localcontext(context):
                c = decimal.Decimal(value=d, context=None)
            return x + 1, a is d, b is d, c is d, a, b, c

        x = torch.ones(2)
        d = decimal.Decimal("1.23")
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x, d)
        self.assertEqual(actual[:4], (x + 1, True, True, True))
        for value in actual[4:]:
            self.assertIs(value, d)

    @parametrize(
        "call",
        [
            lambda: decimal.Decimal(1, None, None),
            lambda: decimal.Decimal("1", value=2),
            lambda: decimal.Decimal(1, unknown=True),
            lambda: decimal.Context(prec=0),
            lambda: decimal.localcontext(decimal.Context(), prec=0),
        ],
    )
    def test_invalid_arguments(self, call):
        def fn(x):
            try:
                call()
            except (TypeError, ValueError):
                return x + 1
            return x - 1

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("value", ["NaN", "sNaN", "Infinity", "-0", "1.2345"])
    def test_context_independent_methods(self, value):
        def fn(x):
            d = decimal.Decimal(value)
            return x + 1, (
                d.is_finite(),
                d.is_infinite(),
                d.is_nan(),
                d.is_snan(),
                d.is_qnan(),
                d.is_signed(),
                d.is_zero(),
                d.copy_abs().as_tuple(),
                d.copy_negate().as_tuple(),
                d.copy_sign(decimal.Decimal(-1)).as_tuple(),
            )

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize(
        "name",
        [
            "sqrt",
            "exp",
            "ln",
            "log10",
            "quantize",
            "normalize",
            "to_integral",
            "to_integral_exact",
            "to_integral_value",
        ],
    )
    def test_contextual_methods(self, name):
        def fn(x):
            with decimal.localcontext(decimal.Context(prec=6)) as context:
                d = decimal.Decimal("1.23456789")
                if name == "quantize":
                    result = d.quantize(decimal.Decimal("0.01"))
                else:
                    result = getattr(d, name)()
            return x + float(result), result, context

        x = torch.ones(2)
        expected = fn(x)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(actual[:2], expected[:2])
        self.assertEqual(str(actual[2]), str(expected[2]))

    @parametrize("op", [float, int, bool])
    @parametrize("value", ["-1.25", "-0", "1E+100", "Infinity", "NaN", "sNaN"])
    def test_numeric_conversion(self, op, value):
        def fn(x):
            try:
                result = op(decimal.Decimal(value))
            except (ValueError, OverflowError):
                return x + 2, None
            return x + 1, result

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    def test_factorial_rejects_decimal(self):
        def fn(x):
            try:
                math.factorial(decimal.Decimal(5))
            except TypeError:
                return x + 1
            return x - 1

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("isclose", [math.isclose, cmath.isclose])
    @parametrize(
        "values",
        [
            ("1.00000001", "1.0"),
            ("1.00000001e-100", "1e-100"),
            ("1.00000001e100", "1e100"),
            ("-0", "0"),
            ("Infinity", "Infinity"),
            ("Infinity", "-Infinity"),
            ("NaN", "NaN"),
        ],
    )
    def test_isclose(self, isclose, values):
        def fn(x):
            a, b = (decimal.Decimal(value) for value in values)
            return (
                x + 1,
                isclose(a, b, rel_tol=1e-8),
                isclose(
                    a=a,
                    b=b,
                    rel_tol=decimal.Decimal("1e-9"),
                    abs_tol=decimal.Decimal(0),
                ),
            )

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    def test_cmath_isclose_complex_operand(self):
        def fn(x):
            d = decimal.Decimal(1)
            return x + 1, cmath.isclose(d, 1 + 0j), cmath.isclose(1 + 1j, d)

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("isclose", [math.isclose, cmath.isclose])
    def test_isclose_decimal_inputs(self, isclose):
        def fn(x, a, b):
            return x + 1, isclose(a, b, rel_tol=1e-8)

        x = torch.ones(2)
        a, b = decimal.Decimal("1.00000001"), decimal.Decimal(1)
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x, a, b), fn(x, a, b)
        )

    def test_isclose_conversion_order(self):
        class Number:
            def __init__(self, name, calls):
                self.name = name
                self.calls = calls

            def __float__(self):
                self.calls.append(self.name)
                return 1.0

        def fn(x, calls):
            result = math.isclose(
                abs_tol=Number("abs_tol", calls),
                b=Number("b", calls),
                rel_tol=Number("rel_tol", calls),
                a=decimal.Decimal(1),
            )
            return x + 1, result

        x = torch.ones(2)
        calls = []
        expected = fn(x, calls)
        self.assertEqual(calls, ["b", "rel_tol", "abs_tol"])
        calls.clear()
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        for _ in range(2):
            self.assertEqual(compiled(x, calls), expected)
            self.assertEqual(calls, ["b", "rel_tol", "abs_tol"])
            calls.clear()

    @parametrize("isclose", [math.isclose, cmath.isclose])
    @parametrize("value", ["sNaN", "-1"])
    def test_isclose_errors(self, isclose, value):
        def fn(x):
            try:
                isclose(
                    decimal.Decimal(1),
                    decimal.Decimal(2),
                    rel_tol=decimal.Decimal(value),
                )
            except ValueError:
                return x + 1
            return x - 1

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("op", [str, repr])
    def test_capitals_guard(self, op):
        def fn(x):
            return x + 1, op(decimal.Decimal("1e-20"))

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        with decimal.localcontext() as context:
            for capitals in (0, 1, 0):
                context.capitals = capitals
                self.assertEqual(compiled(x), fn(x))
        self.assertEqual(counter.frame_count, 2)

    @parametrize(
        "rounding", [decimal.ROUND_DOWN, decimal.ROUND_UP, decimal.ROUND_HALF_EVEN]
    )
    def test_localcontext_arithmetic_and_cache(self, rounding):
        def fn(x):
            base = decimal.Context(prec=5, rounding=rounding)
            with decimal.localcontext(base) as context:
                a = decimal.Decimal(1.1)
                b = decimal.Decimal(3)
                result = ((a**2 + b**2).sqrt(), 2 + a, 2 - a, 2 * a, 2 / b, 2**b)
                text = str(result[0])
                same = decimal.getcontext() is context
            return x + float(result[0]), result, text, same, context, base

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        with decimal.localcontext() as ambient:
            ambient.prec = 2
            ambient.traps[decimal.FloatOperation] = True
            before = ambient.copy()
            expected = fn(x)
            previous = None
            for _ in range(3):
                actual = compiled(x)
                self.assertEqual(actual[:4], expected[:4])
                for actual_ctx, expected_ctx in zip(actual[4:], expected[4:]):
                    self.assertEqual(str(actual_ctx), str(expected_ctx))
                self.assertIsNot(actual[4], actual[5])
                self.assertIsNot(actual[4], previous)
                previous = actual[4]
                actual[4].clear_flags()
                self.assertEqual(str(ambient), str(before))
        self.assertEqual(counter.frame_count, 1)

    def test_default_context_guard(self):
        def fn(x):
            with decimal.localcontext(decimal.Context()):
                result = decimal.Decimal(1) / decimal.Decimal(3)
            return x + float(result), result

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        previous = decimal.DefaultContext.prec
        try:
            for precision in (2, 5, 2):
                decimal.DefaultContext.prec = precision
                self.assertEqual(compiled(x), fn(x))
        finally:
            decimal.DefaultContext.prec = previous
        self.assertEqual(counter.frame_count, 2)

    def test_default_context_trap_guard(self):
        def fn(x):
            try:
                with decimal.localcontext(decimal.Context()):
                    d = decimal.Decimal(1.1)
            except decimal.FloatOperation:
                return x - 1
            return x + float(d)

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        previous = decimal.DefaultContext.traps[decimal.FloatOperation]
        try:
            for enabled in (False, True, False):
                decimal.DefaultContext.traps[decimal.FloatOperation] = enabled
                self.assertEqual(compiled(x), fn(x))
        finally:
            decimal.DefaultContext.traps[decimal.FloatOperation] = previous
        self.assertEqual(counter.frame_count, 2)

    def test_context_state_reconstruction(self):
        def fn(x):
            base = decimal.Context(
                prec=3,
                rounding=decimal.ROUND_UP,
                Emin=-4,
                Emax=4,
                capitals=0,
                clamp=1,
                flags=[decimal.Inexact],
                traps=[],
            )
            with decimal.localcontext(base) as context:
                result = decimal.Decimal("9.99e4") * 10
            return x + 1, base, context, result

        x = torch.ones(2)
        expected = fn(x)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(str(actual[1]), str(expected[1]))
        self.assertEqual(str(actual[2]), str(expected[2]))
        self.assertEqual(actual[3].as_tuple(), expected[3].as_tuple())

    def test_explicit_external_context_is_unsupported_without_mutation(self):
        def fn(x, context):
            return x + 1, decimal.Decimal(1.1, context=context)

        context = decimal.Context()
        before = str(context)
        with self.assertRaises(Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(2), context)
        self.assertEqual(str(context), before)

    def test_nonliteral_argument_does_not_execute_user_code(self):
        calls = []

        class Operand:
            def __float__(self):
                calls.append(1)
                return 1.0

        def fn(x, operand):
            with decimal.localcontext(decimal.Context()):
                result = decimal.Decimal(operand)
            return x + 1, result

        with self.assertRaises(Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(2), Operand())
        self.assertEqual(calls, [])

    def test_cond_branch_local_context(self):
        def positive(x):
            with decimal.localcontext(decimal.Context(prec=3)):
                amount = float(decimal.Decimal(1) / 3)
            return x + amount

        def negative(x):
            return x - 1

        def fn(x, pred):
            return torch.cond(pred, positive, negative, (x,))

        x = torch.ones(2)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        for pred in (torch.tensor(True), torch.tensor(False)):
            self.assertEqual(
                compiled(x, pred), positive(x) if pred.item() else negative(x)
            )

    def test_context_methods_and_copy(self):
        def fn(x):
            context = decimal.Context(prec=4)
            alias = context
            other = context.copy()
            a = context.create_decimal("1.2345")
            b = other.create_decimal_from_float(1.2345)
            context.clear_flags()
            other.clear_traps()
            return x + float(a) + float(b), context, alias, other

        x = torch.ones(2)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        expected = fn(x)
        self.assertEqual(actual[0], expected[0])
        self.assertIs(actual[1], actual[2])
        for actual_ctx, expected_ctx in zip(actual[1:], expected[1:]):
            self.assertEqual(str(actual_ctx), str(expected_ctx))

    def test_context_attribute_mutation_is_unsupported(self):
        def fn(x):
            context = decimal.Context()
            context.prec = 2
            return x + 1

        with self.assertRaises(Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(2))

    def test_nested_contexts(self):
        def fn(x):
            with decimal.localcontext(decimal.Context(prec=3)) as outer:
                a = decimal.Decimal(1) / 3
                with decimal.localcontext(
                    decimal.Context(prec=7), rounding=decimal.ROUND_UP
                ) as inner:
                    b = decimal.Decimal(1) / 3
                c = decimal.Decimal(1) / 3
                restored = decimal.getcontext() is outer
            return x + float(b), a, b, c, outer, inner, restored

        x = torch.ones(2)
        with decimal.localcontext() as ambient:
            before = str(ambient)
            actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
            expected = fn(x)
            self.assertEqual(actual[:4], expected[:4])
            self.assertEqual(
                tuple(map(str, actual[4:6])), tuple(map(str, expected[4:6]))
            )
            self.assertTrue(actual[-1])
            self.assertEqual(str(ambient), before)

    def test_localcontext_decimal_inputs(self):
        def fn(x, a, b):
            with decimal.localcontext(decimal.Context(prec=3)):
                result = (a + b) / 3
            return x + float(result), result

        x = torch.ones(2)
        a, b = decimal.Decimal("1.2345"), decimal.Decimal("2.3456")
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x, a, b), fn(x, a, b)
        )

    def test_nested_trap_restores_outer_context(self):
        def fn(x):
            with decimal.localcontext(decimal.Context(prec=5)) as outer:
                try:
                    with decimal.localcontext(
                        decimal.Context(prec=2, traps=[decimal.Inexact])
                    ):
                        decimal.Decimal(1) / 3
                except decimal.Inexact:
                    result = decimal.Decimal(1) / 3
                restored = decimal.getcontext() is outer
            return x + float(result), result, restored

        x = torch.ones(2)
        with decimal.localcontext() as ambient:
            before = str(ambient)
            self.assertEqual(
                torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x)
            )
            self.assertEqual(str(ambient), before)

    def test_trap_exception_cleanup(self):
        def fn(x):
            try:
                with decimal.localcontext(
                    decimal.Context(prec=2, traps=[decimal.Inexact])
                ):
                    decimal.Decimal(1) / 3
            except decimal.Inexact:
                return x + 1
            return x - 1

        x = torch.ones(2)
        with decimal.localcontext() as ambient:
            before = str(ambient)
            self.assertEqual(
                torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x)
            )
            self.assertEqual(str(ambient), before)

    @parametrize(
        "fn",
        [
            lambda x: (x + 1, decimal.Decimal(1) / 3),
            lambda x: (x + 1, decimal.Decimal(1.1)),
            lambda x: (x + 1, decimal.Decimal("invalid")),
            lambda x: (x + 1, decimal.getcontext()),
            lambda x: (x + 1, decimal.localcontext()),
            lambda x: (x + 1, decimal.Context().flags),
        ],
    )
    def test_unsupported_ambient_operations_do_not_leak(self, fn):
        with decimal.localcontext() as ambient:
            ambient.prec = 2
            ambient.traps[decimal.InvalidOperation] = False
            before = str(ambient)
            with self.assertRaises(Unsupported):
                torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(2))
            self.assertEqual(str(ambient), before)

    @parametrize(
        "operation",
        [
            operator.truediv,
            lambda a, b: decimal.Decimal(1.1),
            lambda a, b: decimal.Decimal("invalid"),
        ],
    )
    def test_ambient_fallback_preserves_flags_on_cache_hits(self, operation):
        def fn(x):
            result = operation(decimal.Decimal(1), decimal.Decimal(3))
            return x + 1, result

        compiled = torch.compile(fn, backend="eager", fullgraph=False)
        x = torch.ones(2)
        with decimal.localcontext() as ambient:
            ambient.traps[decimal.InvalidOperation] = False
            for precision in (2, 6, 2):
                ambient.prec = precision
                ambient.clear_flags()
                expected = fn(x)
                expected_flags = dict(ambient.flags)
                ambient.clear_flags()
                actual = compiled(x)
                self.assertEqual(actual[0], expected[0])
                self.assertEqual(actual[1].as_tuple(), expected[1].as_tuple())
                self.assertEqual(dict(ambient.flags), expected_flags)

    @parametrize("graph_break", [False, True])
    def test_unsupported_inside_localcontext_cleans_up(self, graph_break):
        def fn(x):
            with decimal.localcontext(decimal.Context(prec=2)):
                decimal.Decimal(1) / 3
                if graph_break:
                    torch._dynamo.graph_break()
                y = x + 1
            return y

        with decimal.localcontext() as ambient:
            before = str(ambient)
            with self.assertRaises(Unsupported):
                torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(2))
            self.assertEqual(str(ambient), before)
            self.assertEqual(
                torch.compile(fn, backend="eager", fullgraph=False)(torch.ones(2)),
                torch.full((2,), 2.0),
            )
            self.assertEqual(str(ambient), before)

    def test_explicit_context_constructor(self):
        def fn(x):
            context = decimal.Context(traps=[])
            a = decimal.Decimal(1.1, context)
            b = decimal.Decimal("invalid", context=context)
            return x + float(a), a, b, context

        x = torch.ones(2)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        expected = fn(x)
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1].as_tuple(), expected[1].as_tuple())
        self.assertEqual(actual[2].as_tuple(), expected[2].as_tuple())
        self.assertEqual(str(actual[3]), str(expected[3]))

    @parametrize("nested", [False, True])
    def test_localcontext_manager_reuse(self, nested):
        def fn(x):
            manager = decimal.localcontext(decimal.Context(prec=2))
            with manager:
                a = decimal.Decimal(1) / 3
                if nested:
                    with manager:
                        a = decimal.Decimal(1) / 3
            with manager:
                b = decimal.Decimal(1) / 3
            return x + float(a), b

        x = torch.ones(2)
        with decimal.localcontext() as ambient:
            before = str(ambient)
            compiled = torch.compile(fn, backend="eager", fullgraph=True)
            if nested:
                with self.assertRaises(Unsupported):
                    compiled(x)
            else:
                self.assertEqual(compiled(x), fn(x))
            self.assertEqual(str(ambient), before)

    def test_seeded_arithmetic_matrix(self):
        rng = random.Random(0)
        samples = tuple(
            (
                f"{rng.randint(-99999, 99999)}e{rng.randint(-30, 30)}",
                f"{rng.randint(1, 99999)}e{rng.randint(-30, 30)}",
                rng.randint(1, 15),
            )
            for _ in range(50)
        )

        def fn(x):
            results = []
            contexts = []
            for a, b, precision in samples:
                with decimal.localcontext(
                    decimal.Context(prec=precision, Emin=-20, Emax=20, traps=[])
                ) as context:
                    left, right = decimal.Decimal(a), decimal.Decimal(b)
                    values = (
                        left + right,
                        left - right,
                        left * right,
                        left / right,
                        left**2,
                        right.sqrt(),
                    )
                    results.append(tuple((d.as_tuple(), float(d)) for d in values))
                contexts.append(context)
            return x + 1, results, contexts

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        expected = fn(x)
        with decimal.localcontext() as ambient:
            before = str(ambient)
            for _ in range(2):
                actual = compiled(x)
                self.assertEqual(actual[:2], expected[:2])
                self.assertEqual(list(map(str, actual[2])), list(map(str, expected[2])))
                self.assertEqual(str(ambient), before)
        self.assertEqual(counter.frame_count, 1)


instantiate_parametrized_tests(DecimalTests)


if __name__ == "__main__":
    run_tests()
