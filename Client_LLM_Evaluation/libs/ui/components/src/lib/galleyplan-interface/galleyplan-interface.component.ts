import { Component, HostListener } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
interface Item {
 type: string;
 code: string;
}
type GalleyCell = Item | PartitionCell | null;
type PartitionCell = (Item | null)[][];
@Component({
 selector: 'galleyplan-interface',
 templateUrl: './galleyplan-interface.component.html',
 styleUrls: ['./galleyplan-interface.component.scss'],
 imports: [CommonModule, FormsModule]
})
export class GalleyplanInterfaceComponent {
 items: Item[] = [
   { type: 'Coffee Maker', code: '1001' },
   { type: 'Chiller', code: '1002' },
   { type: 'Microwave', code: '1003' }
 ];
 galleyStructure: GalleyCell[][] = [[null]];  // Initialize with one row and one column
 contextMenuVisible = false;
 contextMenuPosition = { x: '0px', y: '0px' };
 contextMenuRowIndex: number | null = null;
 contextMenuColIndex: number | null = null;
 draggedItem: Item | null = null;
 onDragStart(item: Item) {
   this.draggedItem = item;
 }
 onDrop(row: number, col: number) {
   if (this.draggedItem) {
     if (!this.galleyStructure[row]) {
       this.galleyStructure[row] = [];
     }
     this.galleyStructure[row][col] = this.draggedItem;
     this.draggedItem = null;
   }
 }
 showContextMenu(event: MouseEvent, rowIndex: number, colIndex: number) {
   event.preventDefault();
   event.stopPropagation();
   this.contextMenuVisible = true;
   this.contextMenuPosition = { x: event.clientX + 'px', y: event.clientY + 'px' };
   this.contextMenuRowIndex = rowIndex;
   this.contextMenuColIndex = colIndex;
 }
 hideContextMenu() {
   this.contextMenuVisible = false;
 }
 createRow() {
   this.galleyStructure.push(Array(this.galleyStructure[0].length).fill(null));
   this.hideContextMenu();
 }
 createRowAbove() {
   if (this.contextMenuRowIndex !== null) {
     this.galleyStructure.splice(this.contextMenuRowIndex, 0, Array(this.galleyStructure[0].length).fill(null));
   }
   this.hideContextMenu();
 }
 createRowBelow() {
   if (this.contextMenuRowIndex !== null) {
     this.galleyStructure.splice(this.contextMenuRowIndex + 1, 0, Array(this.galleyStructure[0].length).fill(null));
   }
   this.hideContextMenu();
 }
 createColumn() {
   if (this.contextMenuRowIndex !== null && this.contextMenuColIndex !== null) {
     this.galleyStructure[this.contextMenuRowIndex].splice(this.contextMenuColIndex + 1, 0, null);
   }
   this.hideContextMenu();
 }
 createColumnBefore() {
   if (this.contextMenuRowIndex !== null && this.contextMenuColIndex !== null) {
     this.galleyStructure[this.contextMenuRowIndex].splice(this.contextMenuColIndex, 0, null);
   }
   this.hideContextMenu();
 }
 createColumnAfter() {
   if (this.contextMenuRowIndex !== null && this.contextMenuColIndex !== null) {
     this.galleyStructure[this.contextMenuRowIndex].splice(this.contextMenuColIndex + 1, 0, null);
   }
   this.hideContextMenu();
 }
 createPartition() {
   if (this.contextMenuRowIndex !== null && this.contextMenuColIndex !== null) {
     this.galleyStructure[this.contextMenuRowIndex][this.contextMenuColIndex] = [[]];
   }
   this.hideContextMenu();
 }
 createPartitionRow() {
   if (this.contextMenuRowIndex !== null && this.contextMenuColIndex !== null) {
     const partition = this.galleyStructure[this.contextMenuRowIndex][this.contextMenuColIndex] as PartitionCell;
     if (Array.isArray(partition)) {
       partition.push(Array(partition[0].length).fill(null));
     }
   }
   this.hideContextMenu();
 }
 createPartitionColumn() {
   if (this.contextMenuRowIndex !== null && this.contextMenuColIndex !== null) {
     const partition = this.galleyStructure[this.contextMenuRowIndex][this.contextMenuColIndex] as PartitionCell;
     if (Array.isArray(partition)) {
       for (const row of partition) {
         row.push(null);
       }
     }
   }
   this.hideContextMenu();
 }
 deleteRow() {
   if (this.contextMenuRowIndex !== null) {
     this.galleyStructure.splice(this.contextMenuRowIndex, 1);
   }
   this.hideContextMenu();
 }
 deleteColumn() {
   if (this.contextMenuColIndex !== null) {
     for (const row of this.galleyStructure) {
       row.splice(this.contextMenuColIndex, 1);
     }
   }
   this.hideContextMenu();
 }
 deleteItem() {
   if (this.contextMenuRowIndex !== null && this.contextMenuColIndex !== null) {
     this.galleyStructure[this.contextMenuRowIndex][this.contextMenuColIndex] = null;
   }
   this.hideContextMenu();
 }
 isPartition(cell: GalleyCell): cell is PartitionCell {
   return Array.isArray(cell);
 }
 @HostListener('document:click', ['$event'])
 onDocumentClick(event: MouseEvent) {
   this.hideContextMenu();
 }
}
