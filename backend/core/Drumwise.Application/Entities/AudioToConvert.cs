using Drumwise.Application.Common.Models;

namespace Drumwise.Application.Entities;

public class AudioToConvert : BaseAuditableEntity
{
    public bool Uploaded { get; set; }
    public string? GDAudioFileId { get; set; }
    public string? GDConvertedMidiFileId { get; set; }
    public bool Converted { get; set; }
}

enum Item
{
    Priority,
    Quantity
}