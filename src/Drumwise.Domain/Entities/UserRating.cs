using Drumwise.Domain.Common;

namespace Drumwise.Domain.Entities;

public class UserRating : BaseAuditableEntity
{
    public string? AssignedTo { get; set; }
    public int Mark { get; set; }
    public string? Comment { get; set; }
}