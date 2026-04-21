export function tensorboardUrl(port) {
  const host = window.location.hostname;
  const match = host.match(/^(.+?)-(\d+)(\.proxy\.runpod\.net)$/);
  if (match) return `https://${match[1]}-${port}${match[3]}`;
  return `http://${host}:${port}`;
}
