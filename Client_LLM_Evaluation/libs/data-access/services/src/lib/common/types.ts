

export type PlainObject = {
  [key: string]:
    | string
    | string[]
    | number
    | number[]
    | boolean
    | boolean[]
    | PlainObject
    | PlainObject[]
    | null
    | undefined;
};

export type BodyData =
  | string
  | PlainObject
  | ArrayBuffer
  | ArrayBufferView
  | URLSearchParams
  | FormData
  | File
  | Blob
  | any
  | FormData
  | Record<string, any>;

export type HttpParameter = PlainObject | URLSearchParams;

