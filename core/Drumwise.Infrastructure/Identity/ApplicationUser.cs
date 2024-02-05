using System.ComponentModel.DataAnnotations;
using Microsoft.AspNetCore.Identity;

namespace Drumwise.Infrastructure.Identity;

public class ApplicationUser : IdentityUser
{
    [StringLength(32)]
    public string? Name { get; set; }
    [StringLength(32)]
    public string? Surname { get; set; }
    
    /// <summary>
    /// Years and fraction of year of experience on drums 
    /// </summary>
    public float Experience { get; set; }
}