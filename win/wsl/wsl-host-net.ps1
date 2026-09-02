param(
  [string]$Iface = "vEthernet (WSL)",
  [string]$HostIp = "172.19.16.1",
  [string]$Mask = "255.255.240.0",
  [string]$NatPrefix = "172.19.16.0/20",
  [string]$NatName = "WslFixNat17219"
)

$ErrorActionPreference = "Stop"

Write-Host "[host] set $Iface -> $HostIp / $Mask"
& netsh.exe interface ipv4 set address name="$Iface" source=static addr=$HostIp mask=$Mask gateway=none
if ($LASTEXITCODE -ne 0) { throw "netsh set address failed: $LASTEXITCODE" }

Write-Host "[host] enable IPv4 forwarding on $Iface"
Set-NetIPInterface -InterfaceAlias $Iface -AddressFamily IPv4 -Forwarding Enabled

$def = Get-NetRoute -DestinationPrefix "0.0.0.0/0" -AddressFamily IPv4 |
  Sort-Object { [int]$_.RouteMetric + [int]$_.InterfaceMetric } |
  Select-Object -First 1
if ($null -ne $def) {
  Write-Host "[host] enable IPv4 forwarding on default-route ifIndex $($def.InterfaceIndex)"
  Set-NetIPInterface -InterfaceIndex $def.InterfaceIndex -AddressFamily IPv4 -Forwarding Enabled
}

$nat = Get-NetNat -ErrorAction SilentlyContinue | Where-Object { $_.InternalIPInterfaceAddressPrefix -eq $NatPrefix }
if ($null -eq $nat) {
  Write-Host "[host] create NetNat $NatName for $NatPrefix"
  New-NetNat -Name $NatName -InternalIPInterfaceAddressPrefix $NatPrefix | Out-Null
} else {
  Write-Host "[host] NetNat for $NatPrefix already exists: $($nat.Name)"
}

$rule = "WSL-FixNet-Allow-vEthernet"
if (-not (Get-NetFirewallRule -DisplayName $rule -ErrorAction SilentlyContinue)) {
  Write-Host "[host] add inbound firewall allow on $Iface"
  New-NetFirewallRule -DisplayName $rule -Direction Inbound -InterfaceAlias $Iface -Action Allow | Out-Null
}

Write-Host "[host] done"