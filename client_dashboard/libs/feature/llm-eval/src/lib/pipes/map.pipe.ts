/* Path: libs/feature/llm-eval/src/lib/pipes/map.pipe.ts */
import { Pipe, PipeTransform } from '@angular/core';

/**
 * Map Pipe allows transforming objects in an array using a mapping function
 * Useful for converting complex objects to simpler formats for display
 *
 * Example usage:
 * *ngFor="let item of items | map:'displayName'"
 *   - returns array of item.displayName values
 *
 * *ngFor="let item of items | map:nameMapFn"
 *   - applies the nameMapFn to each item
 */
@Pipe({
  name: 'map',
  standalone: true
})
export class MapPipe implements PipeTransform {
  transform<T, R>(
    items: T[],
    mappingLogic: string | ((item: T) => R)
  ): R[] {
    if (!items || !Array.isArray(items)) {
      return [];
    }

    // If mappingLogic is a string, extract that property from each item
    if (typeof mappingLogic === 'string') {
      return items.map(item => item[mappingLogic as keyof T] as unknown as R);
    }

    // If mappingLogic is a function, apply it to each item
    if (typeof mappingLogic === 'function') {
      return items.map(mappingLogic);
    }

    // Default: return items as is (cast to R[])
    return items as unknown as R[];
  }
}
