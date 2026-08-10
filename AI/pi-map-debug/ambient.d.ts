// 仅供本地类型检查。运行时 pi 会注入 ExtensionAPI，类型 import 会被擦除。
declare module "@earendil-works/pi-coding-agent" {
  export interface ExtensionAPI {
    registerTool(x: unknown): void;
    [k: string]: unknown;
  }
}