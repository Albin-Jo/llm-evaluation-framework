import { Component, OnInit, Input, Output, EventEmitter } from '@angular/core';
import { RichTextEditorAllModule } from '@syncfusion/ej2-angular-richtexteditor'
import { ToolbarService, LinkService, ImageService, HtmlEditorService } from '@syncfusion/ej2-angular-richtexteditor';
import { CommonModule } from '@angular/common';
@Component({
 selector: 'qrac-richtexteditor',
 templateUrl: './qracrichtexteditor.component.html',
 styleUrls: ['./qracrichtexteditor.component.scss'],
 imports: [CommonModule, RichTextEditorAllModule],
 providers: [ToolbarService, LinkService, ImageService, HtmlEditorService]
})
export class QracRichtexteditorComponent implements OnInit {
  public tools: any = {
    items: ['Undo', 'Redo', '|',
        'Bold', 'Italic', 'Underline', 'StrikeThrough', '|',
        'FontName', 'FontSize', 'FontColor', 'BackgroundColor', '|',
        'SubScript', 'SuperScript', '|',
        'LowerCase', 'UpperCase', '|',
        'Formats', 'Alignments', '|', 'OrderedList', 'UnorderedList', '|',
        'Indent', 'Outdent', '|', 'CreateLink',
        'Image', '|', 'ClearFormat', 'Print', 'SourceCode', '|', 'FullScreen']
};
  ngOnInit(): void {
    // any additional initialization
  }
  }
