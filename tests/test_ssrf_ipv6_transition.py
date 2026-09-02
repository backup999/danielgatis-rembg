import ipaddress

import pytest

from rembg.commands.s_command import _is_blocked_ip, _unwrap_ipv6


@pytest.mark.parametrize(
    "addr, embedded",
    [
        # 6to4 wrapping a private v4. On Python 3.11.9 the wrapper alone
        # reports is_private=False, which is the reported bypass.
        ("2002:c0a8:0101::", "192.168.1.1"),
        ("2002:7f00:0001::", "127.0.0.1"),
        ("2002:a9fe:a9fe::", "169.254.169.254"),  # cloud metadata
        ("2002:0a00:0001::", "10.0.0.1"),
        # NAT64.
        ("64:ff9b::c0a8:101", "192.168.1.1"),
        ("64:ff9b::a9fe:a9fe", "169.254.169.254"),
        # IPv4-mapped.
        ("::ffff:127.0.0.1", "127.0.0.1"),
        ("::ffff:169.254.169.254", "169.254.169.254"),
    ],
)
def test_transition_addresses_wrapping_internal_v4_are_blocked(addr, embedded):
    """An internal IPv4 stays blocked however it is wrapped in IPv6."""
    assert _is_blocked_ip(ipaddress.ip_address(addr)) is True


@pytest.mark.parametrize(
    "addr",
    [
        "192.168.1.1",
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "::1",
        "fe80::1",
        "fc00::1",
        "0.0.0.0",
    ],
)
def test_plain_internal_addresses_stay_blocked(addr):
    """The original checks still hold; unwrapping did not weaken them."""
    assert _is_blocked_ip(ipaddress.ip_address(addr)) is True


@pytest.mark.parametrize(
    "addr",
    [
        "8.8.8.8",
        "1.1.1.1",
        "93.184.216.34",
        "2606:4700:4700::1111",
        "2001:4860:4860::8888",
    ],
)
def test_public_addresses_are_allowed(addr):
    """Ordinary public destinations are not caught by the widened check."""
    assert _is_blocked_ip(ipaddress.ip_address(addr)) is False


def test_sixtofour_is_refused_even_when_it_wraps_a_public_v4():
    """The whole 6to4 range is refused, not just the private payloads.

    Python classifies 2002::/16 as private on its own, so a 6to4 address is
    blocked before its payload is consulted. 6to4 is deprecated (RFC 7526), so
    losing it costs nothing and keeps the guard conservative.
    """
    assert _is_blocked_ip(ipaddress.ip_address("2002:0808:0808::")) is True


def test_unwrap_extracts_embedded_v4():
    assert _unwrap_ipv6(
        ipaddress.ip_address("2002:c0a8:0101::")
    ) == ipaddress.ip_address("192.168.1.1")
    assert _unwrap_ipv6(
        ipaddress.ip_address("::ffff:127.0.0.1")
    ) == ipaddress.ip_address("127.0.0.1")
    assert _unwrap_ipv6(
        ipaddress.ip_address("64:ff9b::c0a8:101")
    ) == ipaddress.ip_address("192.168.1.1")


def test_unwrap_passes_through_plain_addresses():
    """A v4 address, or a v6 that embeds nothing, is returned unchanged."""
    for addr in ("8.8.8.8", "2606:4700:4700::1111", "::1"):
        ip = ipaddress.ip_address(addr)
        assert _unwrap_ipv6(ip) == ip
